/* JPEG Crush — CUDA kernels.
 * validated: CUDA 13.3.33; fat binary Turing..Blackwell.
 */
#include "JpegCrushCUDA.h"
#include "CudaCheck.h"

#include "device_launch_parameters.h"
#include <algorithm>

namespace jpegcrush {

static constexpr float RING_K = 3.0f;   // max AC scale swing

// host/device-identical hashes
static __host__ __device__ __forceinline__ uint32_t hashU32(uint32_t x)
{
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static __host__ __device__ __forceinline__ uint32_t hash2(uint32_t a, uint32_t b)
{
    return hashU32((a * 0x9e3779b9U) ^ hashU32(b + 0x165667b1U));
}
static __host__ __device__ __forceinline__ float hf(uint32_t h)
{
    return (float)h * (1.0f / 4294967296.0f) * 2.0f - 1.0f;
}
// per-block, per-coefficient noise
static __host__ __device__ __forceinline__ float acNoise(int bx, int by, int idx, uint32_t seed)
{
    return hf(hash2(((uint32_t)bx * 73856093u) ^ ((uint32_t)by * 19349663u) ^ seed,
                    (uint32_t)idx * 0x9e3779b9u ^ 0xC2B2AE35u));
}
static __host__ __device__ __forceinline__ float roundq(float x)
{
    return (x >= 0.0f) ? floorf(x + 0.5f) : ceilf(x - 0.5f);
}
static __host__ __device__ __forceinline__ uint32_t toByte(float v)
{
    float r = roundq(v);
    if (r < 0.0f) r = 0.0f;
    if (r > 255.0f) r = 255.0f;
    return (uint32_t)r;
}
static __host__ __device__ __forceinline__ int divUp(int a, int b) { return (a + b - 1) / b; }
static __host__ __device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// bypass: copy input to output
__global__ void passthroughKernel(
    cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf, int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    uchar4 c;
    surf2Dread(&c, inSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeClamp);
    surf2Dwrite(c, outSurf, x * (int)sizeof(uchar4), y, cudaBoundaryModeZero);
}

// 1 block per NxN image block, NxN threads. templated on N (8/16/32). tables: [C(NxN)|qLum|qChr]
template <int N>
__global__ void jpegCrushKernel(
    cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf, int W, int H, bool bgra,
    const float* __restrict__ tables, float ringing,
    int subW, int subH, int offX, int offY, uint32_t seedU32)
{
    const float* C    = tables;
    const float* qLum = tables + N * N;
    const float* qChr = tables + 2 * N * N;

    __shared__ float sC[N][N];
    __shared__ float blk[3][N][N];
    __shared__ float tmp[N][N];

    int tx = threadIdx.x, ty = threadIdx.y;     // 0..N-1
    int bx = blockIdx.x,  by = blockIdx.y;
    // shifted-grid image-space pixel
    int px = bx * N + tx - offX, py = by * N + ty - offY;

    sC[ty][tx] = C[ty * N + tx];

    // luma from own pixel (border-clamped)
    int cx = clampi(px, 0, W - 1);
    int cy = clampi(py, 0, H - 1);
    uchar4 c;
    surf2Dread(&c, inSurf, cx * (int)sizeof(uchar4), cy, cudaBoundaryModeClamp);
    float R, G, B, A = (float)c.w;
    if (bgra) { B = c.x; G = c.y; R = c.z; } else { R = c.x; G = c.y; B = c.z; }

    // chroma = box-averaged subsample cell (RGB avg == Cb/Cr avg since affine).
    // strided to ~4x4 samples: real-JPEG modes fully sampled, big cells strided (~64x fewer reads at 32x32)
    float Rc = R, Gc = G, Bc = B;
    if (subW > 1 || subH > 1)
    {
        int sx0 = (cx / subW) * subW;
        int sy0 = (cy / subH) * subH;
        int stepX = (subW + 3) / 4;
        int stepY = (subH + 3) / 4;
        float sr = 0.f, sg = 0.f, sb = 0.f; int n = 0;
        for (int j = 0; j < subH; j += stepY)
            for (int i = 0; i < subW; i += stepX)
            {
                int ux = clampi(sx0 + i, 0, W - 1);
                int uy = clampi(sy0 + j, 0, H - 1);
                uchar4 cc;
                surf2Dread(&cc, inSurf, ux * (int)sizeof(uchar4), uy, cudaBoundaryModeClamp);
                if (bgra) { sb += cc.x; sg += cc.y; sr += cc.z; }
                else      { sr += cc.x; sg += cc.y; sb += cc.z; }
                ++n;
            }
        float inv = 1.0f / (float)n;
        Rc = sr * inv; Gc = sg * inv; Bc = sb * inv;
    }

    float Y  =        0.299f * R  + 0.587f * G  + 0.114f * B;
    float Cb = 128.f - 0.168736f * Rc - 0.331264f * Gc + 0.5f * Bc;
    float Cr = 128.f + 0.5f * Rc - 0.418688f * Gc - 0.081312f * Bc;
    blk[0][ty][tx] = Y  - 128.f;
    blk[1][ty][tx] = Cb - 128.f;
    blk[2][ty][tx] = Cr - 128.f;
    __syncthreads();

    int idx = ty * N + tx;
    for (int ch = 0; ch < 3; ++ch)
    {
        const float* Q = (ch == 0) ? qLum : qChr;

        // forward 2D DCT: F = C X C^T
        float acc = 0.f;                                  // A = C X
        for (int r = 0; r < N; ++r) acc += sC[ty][r] * blk[ch][r][tx];
        __syncthreads(); tmp[ty][tx] = acc; __syncthreads();
        float F = 0.f;                                    // F = A C^T
        for (int cc = 0; cc < N; ++cc) F += tmp[ty][cc] * sC[tx][cc];

        // quantize, then AC ringing
        float qstep = Q[idx];
        float qc = roundq(F / qstep);

        if (idx != 0)
            qc *= (1.0f + ringing * acNoise(bx, by, idx, seedU32) * RING_K);

        float Fd = qc * qstep;

        __syncthreads(); tmp[ty][tx] = Fd; __syncthreads();
        // inverse 2D DCT: X = C^T F C
        float D = 0.f;                                    // D = C^T F
        for (int u = 0; u < N; ++u) D += sC[u][ty] * tmp[u][tx];
        __syncthreads(); tmp[ty][tx] = D; __syncthreads();
        float X = 0.f;                                    // X = D C
        for (int v = 0; v < N; ++v) X += tmp[ty][v] * sC[v][tx];
        blk[ch][ty][tx] = X;
        __syncthreads();
    }

    // YCbCr -> RGB; blk = Y-128, Cb-128, Cr-128
    float Yv = blk[0][ty][tx] + 128.f;
    float cb = blk[1][ty][tx];
    float cr = blk[2][ty][tx];
    float Ro = Yv + 1.402f * cr;
    float Go = Yv - 0.344136f * cb - 0.714136f * cr;
    float Bo = Yv + 1.772f * cb;

    uint32_t rb = toByte(Ro), gb = toByte(Go), bb = toByte(Bo), ab = (uint32_t)A;
    uint32_t out = bgra ? (bb | (gb << 8) | (rb << 16) | (ab << 24))
                        : (rb | (gb << 8) | (bb << 16) | (ab << 24));

    if (px >= 0 && py >= 0 && px < W && py < H)
        surf2Dwrite(out, outSurf, px * (int)sizeof(uchar4), py, cudaBoundaryModeZero);
}

// dispatch templated instantiation for N (8/16/32)
static void launchCrush(int N, dim3 grid, dim3 block, cudaStream_t stream,
    cudaSurfaceObject_t s, cudaSurfaceObject_t d, int W, int H, bool bgra,
    const float* tables, float ringing, int subW, int subH,
    int ox, int oy, uint32_t seed)
{
    if (N == 16)
        jpegCrushKernel<16><<<grid, block, 0, stream>>>(
            s, d, W, H, bgra, tables, ringing, subW, subH, ox, oy, seed);
    else if (N == 32)
        jpegCrushKernel<32><<<grid, block, 0, stream>>>(
            s, d, W, H, bgra, tables, ringing, subW, subH, ox, oy, seed);
    else
        jpegCrushKernel<8><<<grid, block, 0, stream>>>(
            s, d, W, H, bgra, tables, ringing, subW, subH, ox, oy, seed);
}

JpegCrusher::~JpegCrusher()
{
    cudaFree(myTables);
    if (myPingSurf) cudaDestroySurfaceObject(myPingSurf);
    if (myPongSurf) cudaDestroySurfaceObject(myPongSurf);
    if (myPing) cudaFreeArray(myPing);
    if (myPong) cudaFreeArray(myPong);
}

cudaError_t JpegCrusher::ensureInit(const char** outError)
{
    if (myTables) return cudaSuccess;
    // sized for largest block; per-N tables uploaded in process() on change
    BG_CUDA_RETURN(cudaMalloc(&myTables, 3 * kMaxBlockArea * sizeof(float)), outError);
    myLastLumaQ = -1.0f; myLastChromaQ = -1.0f; myLastBlock = -1;  // force first upload
    return cudaSuccess;
}

// ping-pong scratch surfaces for generation passes
cudaError_t JpegCrusher::ensureScratch(int W, int H, const char** outError)
{
    if (myPing && myScratchW == W && myScratchH == H)
        return cudaSuccess;  // reuse

    if (myPingSurf) { cudaDestroySurfaceObject(myPingSurf); myPingSurf = 0; }
    if (myPongSurf) { cudaDestroySurfaceObject(myPongSurf); myPongSurf = 0; }
    if (myPing) { cudaFreeArray(myPing); myPing = nullptr; }
    if (myPong) { cudaFreeArray(myPong); myPong = nullptr; }

    cudaChannelFormatDesc cd = cudaCreateChannelDesc<uchar4>();
    BG_CUDA_RETURN(cudaMallocArray(&myPing, &cd, W, H, cudaArraySurfaceLoadStore), outError);
    BG_CUDA_RETURN(cudaMallocArray(&myPong, &cd, W, H, cudaArraySurfaceLoadStore), outError);

    cudaResourceDesc rd; memset(&rd, 0, sizeof(rd)); rd.resType = cudaResourceTypeArray;
    rd.res.array.array = myPing;
    BG_CUDA_RETURN(cudaCreateSurfaceObject(&myPingSurf, &rd), outError);
    rd.res.array.array = myPong;
    BG_CUDA_RETURN(cudaCreateSurfaceObject(&myPongSurf, &rd), outError);

    myScratchW = W; myScratchH = H;
    return cudaSuccess;
}

cudaError_t JpegCrusher::process(
    cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf,
    const Params& p, cudaStream_t stream, const char** outError)
{
    if (outError) *outError = nullptr;
    const int W = p.width, H = p.height;

    if (!inSurf)
        return cudaSuccess;  // no input: output left as-is

    // bypass
    if (p.bypass)
    {
        dim3 b(16, 16, 1);
        dim3 g(divUp(W, 16), divUp(H, 16), 1);
        passthroughKernel<<<g, b, 0, stream>>>(inSurf, outSurf, W, H);
        BG_CUDA_CHECK_LAUNCH(outError);
        return cudaSuccess;
    }

    BG_CUDA_RETURN(ensureInit(outError), outError);

    int N = p.blockSize;
    if (N != 8 && N != 16 && N != 32) N = 8;

    // re-upload tables only when block size or a quality changes
    if (N != myLastBlock || p.quality != myLastLumaQ || p.chromaQuality != myLastChromaQ)
    {
        computeDctMatrixN(N, myHost);
        computeQuantTablesN(N, p.quality, p.chromaQuality, myHost + N * N, myHost + 2 * N * N);
        BG_CUDA_RETURN(cudaMemcpyAsync(myTables, myHost, 3 * N * N * sizeof(float),
                       cudaMemcpyHostToDevice, stream), outError);
        myLastBlock = N; myLastLumaQ = p.quality; myLastChromaQ = p.chromaQuality;
    }

    uint32_t seedU32 = hashU32((uint32_t)(p.seed * 1000.0f + 0.5f));

    int subW = 1, subH = 1; subsampleCell(p.subsample, subW, subH);

    // +1 block each way to cover shifted grid; OOB threads skip write
    dim3 block(N, N, 1);
    dim3 grid(divUp(W, N) + 1, divUp(H, N) + 1, 1);

    int G = p.generations;
    if (G < 1) G = 1;
    if (G > kMaxGenerations) G = kMaxGenerations;

    if (G == 1)
    {
        launchCrush(N, grid, block, stream, inSurf, outSurf, W, H, p.bgra, myTables,
                    p.ringing, subW, subH, 0, 0, seedU32);
        BG_CUDA_CHECK_LAUNCH(outError);
    }
    else
    {
        // generation loss: ping-pong scratch, shift grid each pass (gen 0 = no shift), last pass -> output
        BG_CUDA_RETURN(ensureScratch(W, H, outError), outError);
        cudaSurfaceObject_t src = inSurf;
        for (int g = 0; g < G; ++g)
        {
            int ox = 0, oy = 0;
            if (g > 0)
            {
                ox = 1 + (int)(hashU32((uint32_t)g * 2654435761u) % 7u);
                oy = 1 + (int)(hashU32((uint32_t)g * 40503u + 17u) % 7u);
            }
            cudaSurfaceObject_t dst = (g == G - 1) ? outSurf
                                    : ((g & 1) ? myPongSurf : myPingSurf);
            launchCrush(N, grid, block, stream, src, dst, W, H, p.bgra, myTables,
                        p.ringing, subW, subH, ox, oy, seedU32);
            src = dst;
        }
        BG_CUDA_CHECK_LAUNCH(outError);
    }

    return cudaSuccess;
}

} // namespace jpegcrush

/* JPEG Crush — CUDA algorithm interface (real-time JPEG/DCT artifact engine).
 *
 * Plain C++17 contract between the TouchDesigner glue (JpegCrushTOP.cpp) and the CUDA
 * implementation (JpegCrushCUDA.cu). Also holds host-side helpers (DCT matrix + JPEG
 * quant tables + shared lookups) used by both the device upload and the test's CPU
 * reference, so both use bit-identical constants (the device never calls cosf).
 *
 * Per NxN block (block size 8/16/32): RGB->YCbCr (JFIF), chroma subsampling (box-averaged),
 * level shift, 2D DCT, quantize (Annex-K tables scaled by separate luma/chroma quality),
 * multiplicative AC ringing, dequantize, inverse DCT, YCbCr->RGB. Generation loss re-runs
 * the whole codec N times on a per-pass-shifted block grid (breaks JPEG idempotency ->
 * visible recompression).
 */
#ifndef JPEGCRUSH_CUDA_H
#define JPEGCRUSH_CUDA_H

#include "cuda_runtime.h"
#include <cstdint>

namespace jpegcrush {

static constexpr int kBlock = 8;                 // standard JPEG block size
static constexpr int kBlockArea = kBlock * kBlock;
static constexpr int kMaxBlock = 32;             // largest selectable transform block
static constexpr int kMaxBlockArea = kMaxBlock * kMaxBlock;

// Upper bound on generation-loss passes (also the slider max).
static constexpr int kMaxGenerations = 16;

struct Params
{
    int32_t  width  = 0;
    int32_t  height = 0;

    // Transform block size in pixels: 8 (standard JPEG), 16, or 32 (modern-codec-style
    // larger transforms -> chunkier blocking + longer-range ringing).
    int32_t  blockSize      = 8;

    // Quantization (1 = near-lossless, 0 = brutal). Luma and chroma are independent so
    // color can be crushed far harder than detail (as real JPEG encoders do).
    float    quality        = 0.5f;
    float    chromaQuality  = 0.5f;

    // Chroma subsampling mode (box-averaged). 0:4:4:4 1:4:2:2 2:4:2:0 3:4:1:1
    // 4:8x8 5:16x16 6:32x32. See subsampleCell().
    int32_t  subsample      = 0;

    float    ringing        = 0.5f; // multiplicative AC corruption: edge ringing
    float    seed           = 0.0f; // varies the Ringing noise pattern

    // Generation loss: recompress this many times on a per-pass-shifted grid. 1 = single
    // compression; higher compounds blocking/ringing/color drift (re-saved-JPEG look).
    int32_t  generations    = 1;    // clamped to [1, kMaxGenerations]

    bool     bgra           = true; // BGRA8 vs RGBA8 channel order
    bool     bypass         = false;// copy input straight through (native-bypass stand-in)
};

// Chroma subsample cell size in pixels for a given mode (box-averaged). Modes 0-3 are the
// real JPEG ratios (color averaged within the 8x8 block). Modes 4-6 use progressively
// larger cells that span MULTIPLE blocks, so color goes flat across several blocks — the
// effect stays visible even when Chroma Quality is low.
static inline __host__ __device__ void subsampleCell(int mode, int& sw, int& sh)
{
    switch (mode)
    {
        case 1: sw = 2;  sh = 1;  break;   // 4:2:2
        case 2: sw = 2;  sh = 2;  break;   // 4:2:0
        case 3: sw = 4;  sh = 1;  break;   // 4:1:1
        case 4: sw = 8;  sh = 8;  break;   // 8x8 cell  (1 block)
        case 5: sw = 16; sh = 16; break;   // 16x16     (2x2 blocks)
        case 6: sw = 32; sh = 32; break;   // 32x32     (4x4 blocks)
        case 0: default: sw = 1; sh = 1; break;  // 4:4:4 (full)
    }
}

// Orthonormal NxN DCT-II matrix C, row-major: forward F = C X C^T, inverse X = C^T F C.
inline void computeDctMatrixN(int N, float* C)
{
    const double PI = 3.14159265358979323846;
    for (int k = 0; k < N; ++k)
        for (int n = 0; n < N; ++n)
        {
            double a = (k == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
            C[k * N + n] = (float)(a * cos((2 * n + 1) * k * PI / (2.0 * N)));
        }
}

// Bilinear sample of an 8x8 base quant table at fractional coords (fx,fy) in [0,7].
inline float sampleBase8(const int base[kBlockArea], float fx, float fy)
{
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 < 7 ? x0 + 1 : 7, y1 = y0 < 7 ? y0 + 1 : 7;
    float tx = fx - x0, ty = fy - y0;
    float a = (float)base[y0 * 8 + x0], b = (float)base[y0 * 8 + x1];
    float c = (float)base[y1 * 8 + x0], d = (float)base[y1 * 8 + x1];
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

// One NxN quant table: the 8x8 Annex-K base bilinearly upsampled to NxN, then scaled by a
// per-channel quality in [0,1] (libjpeg scaling). For N=8 this reproduces the exact table.
inline void computeQuantTableOneN(int N, const int base[kBlockArea], float quality01, float* q)
{
    float qq = quality01 < 0.f ? 0.f : (quality01 > 1.f ? 1.f : quality01);
    int jq = 1 + (int)(qq * 99.0f + 0.5f);          // JPEG quality 1..100
    if (jq < 1) jq = 1; if (jq > 100) jq = 100;
    int S = (jq < 50) ? (5000 / jq) : (200 - 2 * jq);
    float scale = (N > 1) ? (7.0f / (float)(N - 1)) : 0.0f;
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
        {
            float bv = sampleBase8(base, c * scale, r * scale);
            int v = (int)((bv * S + 50.0f) / 100.0f); if (v < 1) v = 1; if (v > 255) v = 255;
            q[r * N + c] = (float)v;
        }
}

// JPEG luma/chroma quant tables for an NxN block, each scaled by its own quality.
inline void computeQuantTablesN(int N, float qualityLuma, float qualityChroma,
                                float* qLum, float* qChr)
{
    static const int baseLum[kBlockArea] = {
        16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55, 14,13,16,24,40,57,69,56,
        14,17,22,29,51,87,80,62, 18,22,37,56,68,109,103,77, 24,35,55,64,81,104,113,92,
        49,64,78,87,103,121,120,101, 72,92,95,98,112,100,103,99 };
    static const int baseChr[kBlockArea] = {
        17,18,24,47,99,99,99,99, 18,21,26,66,99,99,99,99, 24,26,56,99,99,99,99,99,
        47,66,99,99,99,99,99,99, 99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99, 99,99,99,99,99,99,99,99 };
    computeQuantTableOneN(N, baseLum, qualityLuma,   qLum);
    computeQuantTableOneN(N, baseChr, qualityChroma, qChr);
}

class JpegCrusher
{
public:
    JpegCrusher() = default;
    ~JpegCrusher();

    JpegCrusher(const JpegCrusher&) = delete;
    JpegCrusher& operator=(const JpegCrusher&) = delete;

    // Must be called between TOP_Context::beginCUDAOperations()/endCUDAOperations().
    cudaError_t process(cudaSurfaceObject_t inSurf, cudaSurfaceObject_t outSurf,
                        const Params& p, cudaStream_t stream, const char** outError);

private:
    cudaError_t ensureInit(const char** outError);
    cudaError_t ensureScratch(int width, int height, const char** outError);

    // Device copy of [ DCT matrix (NxN) | qLum (NxN) | qChr (NxN) ], sized for the largest
    // block; only the first 3*N*N entries are used for the active block size N.
    float*   myTables   = nullptr;
    float    myHost[3 * kMaxBlockArea];  // host cache (persists for async uploads)
    float    myLastLumaQ   = -1.0f;   // re-upload tables only when block size or a quality changes
    float    myLastChromaQ = -1.0f;
    int      myLastBlock   = -1;

    // Owned ping-pong arrays for multi-generation recompression (only used when
    // generations > 1). Reused across frames; reallocated only on resolution change.
    cudaArray_t         myPing = nullptr,  myPong = nullptr;
    cudaSurfaceObject_t myPingSurf = 0,    myPongSurf = 0;
    int                 myScratchW = 0,    myScratchH = 0;
};

} // namespace jpegcrush

#endif // JPEGCRUSH_CUDA_H

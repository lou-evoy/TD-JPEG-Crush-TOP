/* CUDA error-checking helpers for the Block Glitch TOP.
 *
 * Kernels launch asynchronously, so most errors only surface at a later synchronizing
 * call. We check every synchronous CUDA call inline and cudaGetLastError() after each
 * launch. Nothing aborts: the algorithm layer returns a cudaError_t and a message to
 * the TD glue, which puts the node into a clean error state via getErrorString().
 */
#ifndef BLOCKGLITCH_CUDA_CHECK_H
#define BLOCKGLITCH_CUDA_CHECK_H

#include "cuda_runtime.h"
#include <cstdio>

#define BG_CUDA_RETURN(expr, outErrPtr)                                          \
    do {                                                                         \
        cudaError_t bg_err__ = (expr);                                           \
        if (bg_err__ != cudaSuccess) {                                           \
            bg_setError((outErrPtr), #expr, bg_err__, __FILE__, __LINE__);       \
            return bg_err__;                                                     \
        }                                                                        \
    } while (0)

#define BG_CUDA_CHECK_LAUNCH(outErrPtr)  BG_CUDA_RETURN(cudaGetLastError(), (outErrPtr))

inline char* bg_errorBuffer()
{
    static char buf[512];
    return buf;
}

inline void bg_setError(const char** outErrPtr, const char* expr,
                        cudaError_t err, const char* file, int line)
{
    char* buf = bg_errorBuffer();
    snprintf(buf, 512, "CUDA error %d (%s) at %s:%d -> %s",
             (int)err, cudaGetErrorString(err), file, line, expr);
#ifdef _DEBUG
    fprintf(stderr, "[BlockGlitchTOP] %s\n", buf);
#endif
    if (outErrPtr)
        *outErrPtr = buf;
}

#endif // BLOCKGLITCH_CUDA_CHECK_H

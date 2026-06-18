/* CUDA error-check helpers.
 * async: errors surface at next sync, so check sync calls inline + getLastError after launch.
 */
#ifndef JPEGCRUSH_CUDA_CHECK_H
#define JPEGCRUSH_CUDA_CHECK_H

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
    fprintf(stderr, "[JpegCrushTOP] %s\n", buf);
#endif
    if (outErrPtr)
        *outErrPtr = buf;
}

#endif // JPEGCRUSH_CUDA_CHECK_H

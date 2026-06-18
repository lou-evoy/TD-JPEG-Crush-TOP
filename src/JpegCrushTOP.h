/* JPEG Crush TOP — TouchDesigner Custom Operator (C++ TOP, CUDA execute mode).
 *
 * Thin SDK-facing glue. All image processing lives in JpegCrushCUDA.{h,cu}.
 * Validated against TouchDesigner 2025.32050, TOP C++ API version 12.
 */
#ifndef JPEGCRUSH_TOP_H
#define JPEGCRUSH_TOP_H

#include "TOP_CPlusPlusBase.h"
#include "cuda_runtime.h"
#include "JpegCrushCUDA.h"

using namespace TD;

class JpegCrushTOP : public TOP_CPlusPlusBase
{
public:
    JpegCrushTOP(const OP_NodeInfo* info, TOP_Context* context);
    virtual ~JpegCrushTOP();

    virtual void    getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved1) override;
    virtual void    execute(TOP_Output*, const OP_Inputs*, void* reserved1) override;

    virtual void    getErrorString(OP_String* error, void* reserved1) override;
    virtual void    getInfoPopupString(OP_String* info, void* reserved1) override;

    virtual void    setupParameters(OP_ParameterManager* manager, void* reserved1) override;

private:
    const OP_NodeInfo*  myNodeInfo;
    TOP_Context*        myContext;
    cudaStream_t        myStream;

    cudaSurfaceObject_t myInputSurface;
    cudaSurfaceObject_t myOutputSurface;

    jpegcrush::JpegCrusher  myCrusher;

    const char*         myError;
};

#endif // JPEGCRUSH_TOP_H

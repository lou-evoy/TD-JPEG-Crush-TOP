/* JPEG Crush TOP — TD glue; mirrors the CudaTOP sample. */
#include "JpegCrushTOP.h"

#include <cassert>
#include <cstdio>
#include <algorithm>

// keep in sync with customOPInfo.major/minorVersion below
static const char* kVersion = "1.0.0";

extern "C"
{

DLLEXPORT void
FillTOPPluginInfo(TOP_PluginInfo* info)
{
    if (!info->setAPIVersion(TOPCPlusPlusAPIVersion))
        return;

    info->executeMode = TOP_ExecuteMode::CUDA;

    info->customOPInfo.opType->setString("Jpegcrush");
    info->customOPInfo.opLabel->setString("JPEG Crush");
    info->customOPInfo.opIcon->setString("JPG");
    info->customOPInfo.authorName->setString("SAT");
    info->customOPInfo.authorEmail->setString("levoy@sat.qc.ca");

    info->customOPInfo.minInputs = 1;
    info->customOPInfo.maxInputs = 1;

    info->customOPInfo.majorVersion = 1;
    info->customOPInfo.minorVersion = 0;
}

DLLEXPORT TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
    return new JpegCrushTOP(info, context);
}

DLLEXPORT void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context* context)
{
    delete (JpegCrushTOP*)instance;
}

} // extern "C"

// recreate each cook, never cache: bypass/reactivate frees the cudaArray (stale handle)
static void
setupCudaSurface(cudaSurfaceObject_t* surface, cudaArray_t array)
{
    if (*surface)
    {
        cudaDestroySurfaceObject(*surface);
        *surface = 0;
    }
    cudaResourceDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = array;
    cudaCreateSurfaceObject(surface, &desc);
}

static bool
isSupported8BitRGBA(OP_PixelFormat f)
{
    return f == OP_PixelFormat::BGRA8Fixed || f == OP_PixelFormat::RGBA8Fixed;
}

JpegCrushTOP::JpegCrushTOP(const OP_NodeInfo* info, TOP_Context* context) :
    myNodeInfo(info), myContext(context), myStream(0),
    myInputSurface(0), myOutputSurface(0), myError(nullptr)
{
    cudaStreamCreate(&myStream);
}

JpegCrushTOP::~JpegCrushTOP()
{
    if (myInputSurface)  cudaDestroySurfaceObject(myInputSurface);
    if (myOutputSurface) cudaDestroySurfaceObject(myOutputSurface);
    if (myStream)        cudaStreamDestroy(myStream);
}

void
JpegCrushTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void*)
{
    ginfo->cookEveryFrame = false;
    ginfo->cookEveryFrameIfAsked = false;
    // Version read-only (runs even with no input, when execute early-returns)
    if (inputs) inputs->enablePar("Version", false);
}

void
JpegCrushTOP::getInfoPopupString(OP_String* info, void*)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "JPEG Crush v%s", kVersion);
    info->setString(buf);
}

void
JpegCrushTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
    myError = nullptr;
    inputs->enablePar("Version", false);   // read-only

    if (inputs->getNumInputs() < 1)
    {
        myError = "Connect a TOP to the input.";
        return;
    }

    const OP_TOPInput* topInput = inputs->getInputTOP(0);
    if (!topInput)
    {
        myError = "Input TOP is invalid.";
        return;
    }

    const OP_TextureDesc& inDesc = topInput->textureDesc;
    if (inDesc.texDim != OP_TexDim::e2D)
    {
        myError = "Only 2D textures are supported (no 3D / cube / 2D-array).";
        return;
    }
    if (!isSupported8BitRGBA(inDesc.pixelFormat))
    {
        myError = "Input must be 8-bit RGBA/BGRA (BGRA8Fixed or RGBA8Fixed).";
        return;
    }

    TOP_CUDAOutputInfo info;
    info.textureDesc = inDesc;
    info.stream      = myStream;

    OP_CUDAAcquireInfo acquireInfo;
    acquireInfo.stream = myStream;
    const OP_CUDAArrayInfo* inputArrayInfo = topInput->getCUDAArray(acquireInfo, nullptr);

    const OP_CUDAArrayInfo* outputArrayInfo = output->createCUDAArray(info, nullptr);
    if (!outputArrayInfo)
    {
        myError = "Failed to create output CUDA array.";
        return;
    }

    jpegcrush::Params p;
    p.width         = (int)inDesc.width;
    p.height        = (int)inDesc.height;
    p.bgra          = (inDesc.pixelFormat == OP_PixelFormat::BGRA8Fixed);
    { int bs = std::clamp(inputs->getParInt("Blocksize"), 0, 2);  // 0->8, 1->16, 2->32
      p.blockSize = (bs == 0) ? 8 : (bs == 1) ? 16 : 32; }
    p.quality       = (float)std::clamp(inputs->getParDouble("Quality"), 0.0, 1.0);
    p.chromaQuality = (float)std::clamp(inputs->getParDouble("Chromaquality"), 0.0, 1.0);
    p.subsample     = std::clamp(inputs->getParInt("Subsample"), 0, 6);
    p.ringing       = (float)std::clamp(inputs->getParDouble("Ringing"), 0.0, 1.0);
    p.generations   = std::clamp(inputs->getParInt("Generations"), 1, jpegcrush::kMaxGenerations);
    p.seed          = (float)inputs->getParDouble("Seed");
    p.bypass        = inputs->getParInt("Bypass") != 0;

    if (!myContext->beginCUDAOperations(nullptr))
    {
        myError = "beginCUDAOperations() failed.";
        return;
    }

    setupCudaSurface(&myOutputSurface, outputArrayInfo->cudaArray);
    if (inputArrayInfo && inputArrayInfo->cudaArray)
        setupCudaSurface(&myInputSurface, inputArrayInfo->cudaArray);
    else if (myInputSurface)
    {
        cudaDestroySurfaceObject(myInputSurface);
        myInputSurface = 0;
    }

    cudaGetLastError();   // swallow benign sticky errors from surface (re)creation

    const char* algoError = nullptr;
    myCrusher.process(myInputSurface, myOutputSurface, p, myStream, &algoError);
    if (algoError)
        myError = algoError;

    myContext->endCUDAOperations(nullptr);
}

void
JpegCrushTOP::getErrorString(OP_String* error, void*)
{
    error->setString(myError);
}

static void appendFloat01(OP_ParameterManager* manager, const char* name, const char* label,
                          const char* page, double def)
{
    OP_NumericParameter np(name);
    np.label = label;
    np.page  = page;
    np.defaultValues[0] = def;
    np.minValues[0] = 0.0;  np.maxValues[0] = 1.0;
    np.minSliders[0] = 0.0; np.maxSliders[0] = 1.0;
    np.clampMins[0] = true; np.clampMaxes[0] = true;
    OP_ParAppendResult res = manager->appendFloat(np);
    assert(res == OP_ParAppendResult::Success);
}

void
JpegCrushTOP::setupParameters(OP_ParameterManager* manager, void*)
{
    const char* page = "Crush";

    // Bypass — API doesn't report native Bypass to plugin, so expose our own
    {
        OP_NumericParameter np("Bypass");
        np.label = "Bypass"; np.page = page; np.defaultValues[0] = 0.0;
        OP_ParAppendResult res = manager->appendToggle(np);
        assert(res == OP_ParAppendResult::Success);
    }

    // Block Size — transform block px (8 = JPEG; 16/32 chunkier)
    {
        OP_StringParameter sp("Blocksize");
        sp.label = "Block Size";
        sp.page  = page;
        sp.defaultValue = "B8";
        const char* names[]  = { "B8", "B16", "B32" };
        const char* labels[] = { "8x8 (JPEG)", "16x16", "32x32" };
        OP_ParAppendResult res = manager->appendMenu(sp, 3, names, labels);
        assert(res == OP_ParAppendResult::Success);
    }

    appendFloat01(manager, "Quality",       "Quality",        page, 0.5);
    appendFloat01(manager, "Chromaquality", "Chroma Quality", page, 0.5);

    // chroma subsample (box-averaged)
    {
        OP_StringParameter sp("Subsample");
        sp.label = "Subsample";
        sp.page  = page;
        sp.defaultValue = "S444";
        const char* names[]  = { "S444", "S422", "S420", "S411", "S8", "S16", "S32" };
        const char* labels[] = { "4:4:4 (full)", "4:2:2", "4:2:0", "4:1:1",
                                 "8x8 cell", "16x16 (chunky)", "32x32 (extreme)" };
        OP_ParAppendResult res = manager->appendMenu(sp, 7, names, labels);
        assert(res == OP_ParAppendResult::Success);
    }

    appendFloat01(manager, "Ringing",   "Ringing",    page, 0.5);

    // Generation Loss — recompress N times on a shifted grid
    {
        OP_NumericParameter np("Generations");
        np.label = "Generation Loss";
        np.page  = page;
        np.defaultValues[0] = 1;
        np.minValues[0] = 1;  np.maxValues[0] = (double)jpegcrush::kMaxGenerations;
        np.minSliders[0] = 1; np.maxSliders[0] = (double)jpegcrush::kMaxGenerations;
        np.clampMins[0] = true; np.clampMaxes[0] = true;
        OP_ParAppendResult res = manager->appendInt(np);
        assert(res == OP_ParAppendResult::Success);
    }

    // Seed — Ringing/DC-drift noise pattern (animate for evolving results)
    {
        OP_NumericParameter np("Seed");
        np.label = "Seed";
        np.page  = page;
        np.defaultValues[0] = 0.0;
        np.minValues[0] = 0.0;  np.maxValues[0] = 100.0;
        np.minSliders[0] = 0.0; np.maxSliders[0] = 100.0;
        np.clampMins[0] = true;
        OP_ParAppendResult res = manager->appendFloat(np);
        assert(res == OP_ParAppendResult::Success);
    }

    // Version — read-only
    {
        OP_StringParameter sp("Version");
        sp.label = "Version";
        sp.page  = "Version";
        sp.defaultValue = kVersion;
        OP_ParAppendResult res = manager->appendString(sp);
        assert(res == OP_ParAppendResult::Success);
    }
}

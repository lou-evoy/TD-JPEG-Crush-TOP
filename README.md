# JPEG Crush TOP — real-time JPEG artifact engine for TouchDesigner (CUDA)

A custom TOP that runs an actual JPEG/DCT codec on the input each frame — RGB→YCbCr,
forward DCT, quantization, inverse DCT — so the result is genuine compression artifacts
(8×8 block quantization, ringing, chroma bleed) rather than an imitation, with controls to
drive them well past a normal encoder.

## Demo

<!-- screenshots / GIFs / video go here -->
*Coming soon.*

## Why this one

- **A real codec, not a filter.** The artifacts come from actual coefficient-domain
  quantization, not a "JPEG-look" approximation.
- **Generation loss.** Recompresses the image repeatedly on a shifted block grid,
  accumulating the deep-fried / re-saved-to-death look.
- **Codec controls.** Variable transform block size (8/16/32), independent luma/chroma
  quantization, and chroma subsampling.

## Getting the node

The compiled plugin isn't distributed in this repo. Precompiled builds will be available to
supporters on **Patreon** *(link coming soon)*. If you'd rather compile it yourself, read on.

## Build it yourself

**Requirements:** TouchDesigner 2025.32050 (validated), CUDA Toolkit 12.8+ (13.x validated),
Visual Studio 2022/2026 with the *Desktop development with C++* workload, CMake ≥ 3.24, and an
NVIDIA GPU (Turing / RTX 20-series or newer).

The TD C++ SDK headers (`TOP_CPlusPlusBase.h`, `CPlusPlus_Common.h`) aren't in this repo — they
ship inside TouchDesigner at `<TD install>/Samples/CPlusPlus/CudaTOP`, and `-DTD_SDK_DIR` must
point there (the default assumes a standard `C:/Program Files/Derivative` install).

Run the build from the **x64 Native Tools Command Prompt for VS** (Start menu) — a plain
PowerShell/cmd won't have `cl` and `nvcc` on `PATH`:

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DTD_SDK_DIR="C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
cmake --build build
```

This produces `build/JpegCrushTOP.dll`. Copy it to `%USERPROFILE%\Documents\Derivative\Plugins\`
(or run `cmake --build build --target install_to_td` to do that in one step), restart
TouchDesigner, and add the node from **OP Create → Custom → "JPEG Crush"**.

The build defaults to `sm_75`–`sm_120` and needs CUDA 12.8+ for `sm_120`. On an older toolkit or
GPU, override `PS_CUDA_ARCHITECTURES`, e.g. `-DPS_CUDA_ARCHITECTURES="75-real;86-real;89-real"`
(`nvcc --list-gpu-code` lists what your toolkit supports).

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

**Prerequisites**

- TouchDesigner 2025.30000+
- CUDA Toolkit 13.x (12.8+ for Blackwell / RTX 50)
- Visual Studio 2022 or 2026 (MSVC, *Desktop development with C++*)
- CMake ≥ 3.24

**Build (Release)**

From an *x64 Native Tools Command Prompt* (so `cl` and `nvcc` are on `PATH`):

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
      -DTD_SDK_DIR="C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
cmake --build build
```

Output: `build/JpegCrushTOP.dll`. Copy it to `%USERPROFILE%\Documents\Derivative\Plugins\`,
restart TouchDesigner, and add the node from **OP Create → TOP → "JPEG Crush"**.

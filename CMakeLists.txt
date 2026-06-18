# =============================================================================
# JPEG Crush TOP — TouchDesigner Custom Operator (CUDA, JPEG/DCT artifacts)
# Validated: TouchDesigner 2025.32050 (TOP API v12), CUDA 13.3, VS 2022/2026, CMake >= 3.24.
# =============================================================================
cmake_minimum_required(VERSION 3.24)

project(JpegCrushTOP LANGUAGES CXX CUDA)

set(TD_SDK_DIR
    "C:/Program Files/Derivative/TouchDesigner/Samples/CPlusPlus/CudaTOP"
    CACHE PATH "Folder containing TOP_CPlusPlusBase.h and CPlusPlus_Common.h")

option(PS_ALLOW_UNSUPPORTED_COMPILER "Pass -allow-unsupported-compiler to nvcc" ON)

if(NOT EXISTS "${TD_SDK_DIR}/TOP_CPlusPlusBase.h")
    message(FATAL_ERROR "TOP_CPlusPlusBase.h not found in TD_SDK_DIR='${TD_SDK_DIR}'.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(PS_CUDA_ARCHITECTURES "75-real;86-real;89-real;120-real;120-virtual"
    CACHE STRING "GPU architectures to embed (CMake CUDA_ARCHITECTURES syntax)")

find_package(CUDAToolkit REQUIRED)

add_library(JpegCrushTOP SHARED
    src/JpegCrushTOP.cpp
    src/JpegCrushCUDA.cu
)

target_include_directories(JpegCrushTOP PRIVATE
    "${TD_SDK_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(JpegCrushTOP PRIVATE CUDA::cudart_static)

set_target_properties(JpegCrushTOP PROPERTIES
    PREFIX ""
    OUTPUT_NAME "JpegCrushTOP"
    CUDA_ARCHITECTURES "${PS_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Static
)

set_property(TARGET JpegCrushTOP PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

target_compile_definitions(JpegCrushTOP PRIVATE
    $<$<CONFIG:Debug>:_DEBUG>
    WIN32_LEAN_AND_MEAN
)

target_compile_options(JpegCrushTOP PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
)
if(PS_ALLOW_UNSUPPORTED_COMPILER)
    target_compile_options(JpegCrushTOP PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)
endif()

if(DEFINED ENV{USERPROFILE})
    set(TD_PLUGIN_DIR "$ENV{USERPROFILE}/Documents/Derivative/Plugins")
    add_custom_target(install_to_td
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TD_PLUGIN_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:JpegCrushTOP>" "${TD_PLUGIN_DIR}/"
        DEPENDS JpegCrushTOP
        COMMENT "Copying JpegCrushTOP.dll to ${TD_PLUGIN_DIR}")
endif()

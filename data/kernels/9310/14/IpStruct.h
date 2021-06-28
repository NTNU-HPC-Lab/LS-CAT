#pragma once

#include "Ip/IpType.h"

namespace ip
{

struct roi_t
{
    int x;
    int y;
    int width;
    int height;
};

template <typename TPixel>
struct cpuImage_t
{
    PixelType pixelType;
    int width;
    int height;
    int channels;
    int pixelSize;
    int depth;
    int pitch;
    TPixel* pixels;
    roi_t roi;
};

template <typename TPixel>
struct gpuImage_t
{
    PixelType pixelType;
    int width;
    int height;
    int channels;
    int pitch;
    int pixelSize;
    int depth;
    roi_t roi;
    TPixel* pixels;
};

typedef cpuImage_t<u8>  cpuImageU8_t;
typedef cpuImage_t<u16> cpuImageU16_t;
typedef cpuImage_t<u32> cpuImageU32_t;
typedef cpuImage_t<f32> cpuImageF32_t;
typedef cpuImage_t<f64> cpuImageF64_t;

typedef gpuImage_t<u8>  gpuImageU8_t;
typedef gpuImage_t<u16> gpuImageU16_t;
typedef gpuImage_t<u32> gpuImageU32_t;
typedef gpuImage_t<f32> gpuImageF32_t;
typedef gpuImage_t<f64> gpuImageF64_t;




template <typename TData>
struct cpuArray_t
{
    int size;
    TData* elements;
};

template <typename TData>
struct gpuArray_t
{
    int size;
    TData* elements;
};

typedef cpuArray_t<s8>  cpuArrayS8_t;
typedef cpuArray_t<s16> cpuArrayS16_t;
typedef cpuArray_t<s32> cpuArrayS32_t;
typedef cpuArray_t<s64> cpuArrayS64_t;
typedef cpuArray_t<u8>  cpuArrayU8_t;
typedef cpuArray_t<u16> cpuArrayU16_t;
typedef cpuArray_t<u32> cpuArrayU32_t;
typedef cpuArray_t<u64> cpuArrayU64_t;
typedef cpuArray_t<f32> cpuArrayF32_t;
typedef cpuArray_t<f64> cpuArrayF64_t;

typedef gpuArray_t<s8>  gpuArrayS8_t;
typedef gpuArray_t<s16> gpuArrayS16_t;
typedef gpuArray_t<s32> gpuArrayS32_t;
typedef gpuArray_t<s64> gpuArrayS64_t;
typedef gpuArray_t<u8>  gpuArrayU8_t;
typedef gpuArray_t<u16> gpuArrayU16_t;
typedef gpuArray_t<u32> gpuArrayU32_t;
typedef gpuArray_t<u64> gpuArrayU64_t;
typedef gpuArray_t<f32> gpuArrayF32_t;
typedef gpuArray_t<f64> gpuArrayF64_t;

} // namespace ip


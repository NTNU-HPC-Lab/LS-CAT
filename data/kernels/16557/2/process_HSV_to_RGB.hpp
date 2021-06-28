//
// Created by jlebas01 on 04/04/2020.
//

#ifndef HISTOGRAM_PROJECT_PROCESS_HSV_TO_RGB_HPP
#define HISTOGRAM_PROJECT_PROCESS_HSV_TO_RGB_HPP

#include <vector>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>

namespace process {

    texture<float4, cudaTextureType2D, cudaReadModeElementType> ImgHSV;



    void processHSV_to_RGB(const std::vector<float4> &inputImg, // Input image
                           uint imgWidth, uint imgHeight, // Image size
                           std::vector<uchar4> &output);

    __device__ float4 fRGB_from_HSV(float h, float s, float v, float a);

    __global__ void
    HSV_to_RGB(const size_t imgWidth, const size_t imgHeight, const float *cdf, uchar4 *output);

    __global__ void calculateHistogram(unsigned int *Histogram, size_t width, size_t height);

    __global__ void calcCDFnormalized(const unsigned int *histo, float *cdf, size_t width, size_t height);

    __global__ void calcCDF(float *cdf, unsigned int *histo, int imageWidth, int imageHeight, int length);
}
#endif //HISTOGRAM_PROJECT_PROCESS_HPP

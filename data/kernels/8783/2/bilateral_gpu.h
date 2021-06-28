#pragma once

#ifdef __cplusplus
extern "C"
{
#endif //__cplusplus

void bilateralNaiveGpu(float* inputImage, float* outputImage, int rows, int cols, uint32_t window, float sigmaD, float sigmaR);
void bilateralOptimizedGpu(float* inputImage, float* outputImage, int rows, int cols, uint32_t window, float sigmaD, float sigmaR);

#ifdef __cplusplus
}
#endif //__cplusplus
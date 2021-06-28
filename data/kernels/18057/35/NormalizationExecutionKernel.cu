#include "includes.h"
__global__ void NormalizationExecutionKernel(unsigned char* src, float* dst, const int size, const float alpha, const float beta, const float bias)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
if(index < size){
dst[index] = (float)(src[index] - alpha) / beta + bias;
}
}
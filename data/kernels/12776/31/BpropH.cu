#include "includes.h"
__global__ void BpropH(const float* layer1, float* dlayer1, const float* synH, float* dsynH, const float alpha, const int offset)
{
int i = blockDim.x*blockIdx.x + threadIdx.x; //256
int j = blockDim.y*blockIdx.y + threadIdx.y; //256

atomicAdd(&dsynH[i*256 + j] , dlayer1[offset*256 + j] * layer1[(offset-1)*256 + i] * alpha);
atomicAdd(&dlayer1[(offset-1)*256 + i] , layer1[offset*256 + j] * synH[i*256 + j]);
}
#include "includes.h"
__global__ void FpropH(float* layer1, const float* synH, const int offset)
{
int i = blockDim.x*blockIdx.x + threadIdx.x; //256
int j = blockDim.y*blockIdx.y + threadIdx.y; //256
atomicAdd(&layer1[256*offset + j], layer1[256*(offset-1) + i] * synH[i*256 + j]);
//__syncthreads();
//if (i == 0)
//   layerH[j] = layer1[j];
}
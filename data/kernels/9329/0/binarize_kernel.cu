#include "includes.h"

#ifdef CUDNN
#pragma comment(lib, "cudnn.lib")
#endif

extern "C" {
}


__global__ void binarize_kernel(float *x, int n, float *binary)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i >= n) return;
binary[i] = (x[i] >= 0) ? 1 : -1;
}
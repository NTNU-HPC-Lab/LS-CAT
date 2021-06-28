#include "includes.h"
__global__  void MatrixCopy(float* in, float *out, int size)
{
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
if (id < size)
out[id] = in[id];
}
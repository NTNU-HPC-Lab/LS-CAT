#include "includes.h"

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

__global__ void sum_grad(float* in, int inSize0, int inSize1, int inSize2, int inSize3, int nElement, float* out, int outStride0, int outStride1, int outStride2, int dim) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int i = tid; i < nElement; i += stride) {
int inOff2 = i / inSize3;
int inDim3 = i - inOff2 * inSize3;
int inOff1 = inOff2 / inSize2;
int inDim2 = inOff2 - inOff1 * inSize2;
int inDim0 = inOff1 / inSize1;
int inDim1 = inOff1 - inDim0 * inSize1;
int outOff = 0;
if (dim == 0) outOff = inDim1 * outStride0 + inDim2 * outStride1 + inDim3 * outStride2;
if (dim == 1) outOff = inDim0 * outStride0 + inDim2 * outStride1 + inDim3 * outStride2;
if (dim == 2) outOff = inDim0 * outStride0 + inDim1 * outStride1 + inDim3 * outStride2;
if (dim == 3) outOff = inDim0 * outStride0 + inDim1 * outStride1 + inDim2 * outStride2;
in[i] += out[outOff];
}
}
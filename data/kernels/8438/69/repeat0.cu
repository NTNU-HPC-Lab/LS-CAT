#include "includes.h"
__global__ void repeat0(float* in, float* out, int outStride0, int outStride1, int outScalarCount) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (; tid < outScalarCount; tid += stride) {
int linearIndex = tid;
int outIndex0 = linearIndex / outStride0;
linearIndex = linearIndex - outIndex0 * outStride0;
int outIndex1 = linearIndex / outStride1;
int outIndex2 = linearIndex - outIndex1 * outStride1;
int inIndex = outIndex2 + (outIndex0 + outIndex1) * outStride1;
out[tid] = in[inIndex];
}
}
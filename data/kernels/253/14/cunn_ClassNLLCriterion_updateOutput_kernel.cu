#include "includes.h"
__global__ void cunn_ClassNLLCriterion_updateOutput_kernel(float *output, float *input, float *target, int nframe, int ndim, int sizeAverage, int ntarget) {
__shared__ float shInputs[NTHREADS];
// Verify whether `register` does anything here.
register int i, j, t;

shInputs[threadIdx.x] = .0;
for (i = threadIdx.x; i < nframe; i += NTHREADS) {
for (j = 0; j < ntarget; ++j) {
t = (int)target[i * ntarget + j] - 1;
if (t >= 0)
shInputs[threadIdx.x] += input[i * ndim + t];
}
}
__syncthreads();

// TODO: T4951791 Reuse code between updateOutput_kernel1 and
// updateOutput_kernel
if (threadIdx.x == 0) {
*output = .0;
for (i = 0; i < NTHREADS; ++i)
*output += shInputs[i];
if (sizeAverage)
*output /= nframe;
*output = -(*output);
}
}
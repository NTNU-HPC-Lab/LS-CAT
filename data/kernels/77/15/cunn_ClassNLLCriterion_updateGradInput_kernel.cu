#include "includes.h"
__global__ void cunn_ClassNLLCriterion_updateGradInput_kernel(float *gradInput, float *target, int nframe, int ndim, float grad, int ntarget) {
register int i, j, t;
for (i = threadIdx.x; i < nframe; i += NTHREADS) {
for (j = 0; j < ntarget; ++j) {
t = (int)target[i * ntarget + j] - 1;
if (t >= 0)
gradInput[i * ndim + t] = grad;
}
}
}
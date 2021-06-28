#include "includes.h"


static const int NTHREADS = 32;





__global__ void cunn_ClassNLLCriterion_updateGradInput_kernel( float *gradInput, float *target, float *weights, float *total_weight, int size_average, int nframe, int ndim, int n_classes)
{
if (*total_weight <= 0) {
return;
}
int i, t;
float norm = size_average ? (1.0f / *total_weight) : 1.0f;

for (i = threadIdx.x; i < nframe; i += NTHREADS) {
t = (int)target[i] - 1;
if (t >= 0 && t < n_classes) {
gradInput[i * ndim + t] = -(weights ? weights[t] : 1.0f) * norm;
}
}
}
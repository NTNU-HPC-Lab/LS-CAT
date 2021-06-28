#include "includes.h"
__global__ void lots_of_float_compute(float *inputs, int N, size_t niters, float *outputs)
{
size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
size_t nthreads = gridDim.x * blockDim.x;

for (; tid < N; tid += nthreads)
{
size_t iter;
float val = inputs[tid];

for (iter = 0; iter < niters; iter++)
{
val = (val + 5.0f) - 101.0f;
val = (val / 3.0f) + 102.0f;
val = (val + 1.07f) - 103.0f;
val = (val / 1.037f) + 104.0f;
val = (val + 3.00f) - 105.0f;
val = (val / 0.22f) + 106.0f;
}

outputs[tid] = val;
}
}
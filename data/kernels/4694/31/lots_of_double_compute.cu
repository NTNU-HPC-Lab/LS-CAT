#include "includes.h"
__global__ void lots_of_double_compute(double *inputs, int N, size_t niters, double *outputs)
{
size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
size_t nthreads = gridDim.x * blockDim.x;

for (; tid < N; tid += nthreads)
{
size_t iter;
double val = inputs[tid];

for (iter = 0; iter < niters; iter++)
{
val = (val + 5.0) - 101.0;
val = (val / 3.0) + 102.0;
val = (val + 1.07) - 103.0;
val = (val / 1.037) + 104.0;
val = (val + 3.00) - 105.0;
val = (val / 0.22) + 106.0;
}

outputs[tid] = val;
}
}
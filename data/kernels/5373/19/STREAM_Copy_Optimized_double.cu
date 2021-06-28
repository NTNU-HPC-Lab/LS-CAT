#include "includes.h"
__global__ void STREAM_Copy_Optimized_double(double *a, double *b, size_t len)
{
/*
* Ensure size of thread index space is as large as or greater than
* vector index space else return.
*/
if (blockDim.x * gridDim.x < len) return;
size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < len) b[idx] = a[idx];
}
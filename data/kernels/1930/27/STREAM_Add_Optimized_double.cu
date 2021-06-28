#include "includes.h"
__global__ void STREAM_Add_Optimized_double(double *a, double *b, double *c,  size_t len)
{
/*
* Ensure size of thread index space is as large as or greater than
* vector index space else return.
*/
if (blockDim.x * gridDim.x < len) return;
size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < len) c[idx] = a[idx]+b[idx];
}
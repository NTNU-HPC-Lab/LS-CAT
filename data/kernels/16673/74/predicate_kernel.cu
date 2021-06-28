#include "includes.h"
__global__ void predicate_kernel(float *d_predicates, float *d_input, int length)
{
int idx = blockDim.x * blockIdx.x + threadIdx.x;

if (idx >= length) return;

d_predicates[idx] = d_input[idx] > FLT_ZERO;
}
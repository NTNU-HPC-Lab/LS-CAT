#include "includes.h"
__global__ void pack_kernel(float *d_output, float *d_input, float *d_predicates, float *d_scanned, int length)
{
int idx = blockDim.x * blockIdx.x + threadIdx.x;

if (idx >= length) return;

if (d_predicates[idx] != 0.f)
{
// address
int address = d_scanned[idx] - 1;

// gather
d_output[address] = d_input[idx];
}
}
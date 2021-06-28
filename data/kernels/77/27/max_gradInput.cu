#include "includes.h"
__global__ void max_gradInput(float *input, float *output, float *indices, long nrows, long ncols)
{
// output offset:
long o = threadIdx.x + blockDim.x * blockIdx.x;
if (o >= nrows) return;

// input offset:
long i = o * ncols;

// bprop max gradient:
long idx = indices[o]-1;
input[i+idx] = output[o];
}
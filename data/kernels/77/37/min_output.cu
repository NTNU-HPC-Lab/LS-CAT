#include "includes.h"
__global__ void min_output(float *input, float *output, float *indices, long nrows, long ncols)
{
// output offset:
long o = threadIdx.x + blockDim.x * blockIdx.x;
if (o >= nrows) return;

// input offset:
long i = o * ncols;

// move pointers
input = input + i;

// compute min:
float min = input[0];
long argmin = 0;
long ii;
for (ii=1; ii<ncols; ii++) {
float val = input[ii];
if (val < min) {
min = val;
argmin = ii;
}
}

// store
output[o] = min;
indices[o] = argmin+1;
}
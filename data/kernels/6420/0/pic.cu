#include "includes.h"

#define ITER  10000000000  // Number of bins
#define NUMBLOCKS   13  // Number of thread blocks
#define NUMTHREADS 192  // Number of threads per block
int tid;
float pi;

// Kernel

// Main
__global__ void pic(float *sum, int nbin, float step, int nthreads, int nblocks) {
int i;
float x;
int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
for (i=idx; i< nbin; i+=nthreads*nblocks) {
x = (i+0.5)*step;
sum[idx] += 4.0/(1.0+x*x);
}
}
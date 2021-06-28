#include "includes.h"


// Calculate d[n] = a[n]*b[n] + c[n]

__global__ void custom_kernel(float *a, float *b, float *c, float *d, int N) {
int idx = blockDim.x*blockIdx.x + threadIdx.x;
int num_threads = blockDim.x * gridDim.x;
while(idx < N) {
d[idx] = a[idx]*b[idx]+c[idx];
idx += num_threads;
}
}
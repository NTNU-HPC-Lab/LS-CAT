#include "includes.h"
extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"


extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

extern "C"

/*
* Perfom a reduction from data of length 'size' to result, where length of result will be 'number of blocks'.
*/
extern "C"
__global__ void reducePartial(int size, void *data, void *result) {
float *fdata = (float*) data;
float *sum = (float*) result;

extern __shared__ float sdata[];

// perform first level of reduction,
// reading from global memory, writing to shared memory unsigned int tid = threadIdx.x;
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
sdata[tid] = (i < size ? fdata[i] : 0) + (i+blockDim.x < size ? fdata[i+blockDim.x] : 0);
__syncthreads();

// do reduction in shared mem
for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
if (tid < s) {
sdata[tid] += sdata[tid + s];
}
__syncthreads();
}

// write result for this block to global mem
if (tid == 0) sum[blockIdx.x] = sdata[0];
}
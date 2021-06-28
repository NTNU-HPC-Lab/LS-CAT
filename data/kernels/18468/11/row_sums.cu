#include "includes.h"
__global__ void row_sums(const float *A, float *sums, size_t ds){

int idx = blockIdx.x; // our block index becomes our row indicator
if (idx < ds){
__shared__ float sdata[block_size];
int tid = threadIdx.x;
sdata[tid] = 0.0f;
size_t tidx = tid;

while (tidx < ds) {  // block stride loop to load data
sdata[tid] += A[idx*ds+tidx];
tidx += blockDim.x;
}

for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
__syncthreads();
if (tid < s)  // parallel sweep reduction
sdata[tid] += sdata[tid + s];
}
if (tid == 0) sums[idx] = sdata[0];
}
}
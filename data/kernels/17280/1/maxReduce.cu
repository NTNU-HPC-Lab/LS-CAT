#include "includes.h"
__global__ void maxReduce(int *d_idata, int *d_odata) {
__shared__ int sdata[512];

unsigned int tid = threadIdx.x;
unsigned int index = (blockIdx.x * blockDim.x) + tid;
sdata[tid] = d_idata[index];
__syncthreads();

for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
if (tid < stride){
sdata[tid] = max(sdata[tid], sdata[tid + stride]);
}
}
__syncthreads();

if (tid == 0){
d_odata[blockIdx.x] = sdata[0];
}
}
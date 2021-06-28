#include "includes.h"

//kernel for computing histogram right in memory

//computer partial histogram on shared memory and mix them on global memory

__global__ void hist_inShared (const int* values, int length, int* hist){

//load shared memory
extern __shared__ int shHist[];
shHist[threadIdx.x] = 0;
__syncthreads();

//compute index and interval
int idx = blockDim.x * blockIdx.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;

//iterate over index and interval since it is less than the total length
while(idx < length){
int val = values[idx];
//increment value frequency on histogram using atomic in order to be thread safe
atomicAdd(&shHist[val], 1);
idx += stride;
}

//combine partial histogram on shared memory to create a full histogram
__syncthreads();
atomicAdd(&hist[threadIdx.x], shHist[threadIdx.x]);
}
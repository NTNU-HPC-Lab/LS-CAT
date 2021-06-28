#include "includes.h"

//kernel for computing histogram right in memory

//computer partial histogram on shared memory and mix them on global memory

__global__ void hist_inGlobal (const int* values, int length, int* hist){

//compute index and interval
int idx = blockDim.x * blockIdx.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
//iterate over index and interval since it is less than the total length
while(idx < length){
//get value
int val = values[idx];
//increment value frequency on histogram using atomic in order to be thread safe
atomicAdd(&hist[val], 1);
idx += stride;
}
}
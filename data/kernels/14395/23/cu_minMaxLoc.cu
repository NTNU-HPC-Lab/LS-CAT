#include "includes.h"
__global__ void cu_minMaxLoc(const float* src, float* minValue, float* maxValue, int* minLoc, int* maxLoc, float* minValCache, float* maxValCache, int*   minLocCache, int*   maxLocCache, const int n){
int tid = threadIdx.x + blockIdx.x * blockDim.x;
//int stride = blockDim.x * gridDim.x;
float val = src[0];
int loc = 0;
if(tid < n){
val = src[tid];
loc = tid;
}
maxValCache[threadIdx.x] = val;
minValCache[threadIdx.x] = val;
maxLocCache[threadIdx.x] = loc;
minLocCache[threadIdx.x] = loc;
__syncthreads();
// contiguous range pattern
for(int offset = blockDim.x / 2; offset > 0; offset >>= 1){
if(threadIdx.x < offset){
// add a partial sum upstream to our own
if(maxValCache[threadIdx.x] >= maxValCache[threadIdx.x + offset]){
;
}else{
maxValCache[threadIdx.x] = maxValCache[threadIdx.x + offset];
maxLocCache[threadIdx.x] = maxLocCache[threadIdx.x + offset];
}
if(minValCache[threadIdx.x] <= minValCache[threadIdx.x + offset]){
;
}else{
minValCache[threadIdx.x] = minValCache[threadIdx.x + offset];
minLocCache[threadIdx.x] = minLocCache[threadIdx.x + offset];
}
}
// wait until all threads in the block have
// updated their partial sums
__syncthreads();
}
// thread 0 writes the final result
if(threadIdx.x == 0){
minValue[blockIdx.x] = minValCache[0];
maxValue[blockIdx.x] = maxValCache[0];
minLoc[blockIdx.x] = minLocCache[0];
maxLoc[blockIdx.x] = maxLocCache[0];
}
}
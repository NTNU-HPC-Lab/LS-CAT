#include "includes.h"
__device__ unsigned int getTid3d3d(){
return blockDim.x * ( blockDim.y * ( blockDim.z + ( threadIdx.z * blockDim.y ) )  + threadIdx.y )  + threadIdx.x;
}
__device__ unsigned int getBid3d3d(){
return blockIdx.x + gridDim.x*(blockIdx.y + gridDim.y * blockIdx.z);
}
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__device__ double2 mult(double2 a, double2 b){
return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
}
__device__ double2 mult(double2 a, double b){
return {a.x*b, a.y*b};
}
__global__ void multipass(double* input, double* output){
unsigned int tid = threadIdx.x + threadIdx.y*blockDim.x
+ threadIdx.z * blockDim.x * blockDim.y;
unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;

//unsigned int tid = getTid3d3d();
//unsigned int bid = getBid3d3d();
// printf("bid0=%d\n",bid);

unsigned int gid = getGid3d3d();
extern __shared__ double sdatad[];
sdatad[tid] = input[gid];
__syncthreads();

for(int i = blockDim.x>>1; i > 0; i>>=1){
if(tid < i){
sdatad[tid] += sdatad[tid + i];
}
__syncthreads();
}
if(tid==0){
output[bid] = sdatad[0];
}
}
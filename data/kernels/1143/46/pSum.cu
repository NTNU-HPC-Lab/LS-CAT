#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void pSum(double* in1, double* output, int pass){
unsigned int tid = threadIdx.x;
unsigned int bid = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x;// printf("bid0=%d\n",bid);
unsigned int gid = getGid3d3d();
extern __shared__ double sdata2[];
for(int i = blockDim.x>>1; i > 0; i>>=1){
if(tid < blockDim.x>>1){
sdata2[tid] += sdata2[tid + i];
}
__syncthreads();
}
if(tid==0){
output[bid] = sdata2[0];
}
}
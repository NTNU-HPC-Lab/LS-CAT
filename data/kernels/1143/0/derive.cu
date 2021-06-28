#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void derive(double *data, double *out, int stride, int gsize, double dx){
int gid = getGid3d3d();
if (gid < gsize){
if (gid + stride < gsize){
out[gid] = (data[gid+stride] - data[gid])/dx;
}
else{
out[gid] = data[gid]/dx;
}
}
}
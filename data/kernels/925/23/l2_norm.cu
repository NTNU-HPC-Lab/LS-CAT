#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void l2_norm(double2 *in1, double2 *in2, double *out){

int gid = getGid3d3d();
out[gid] = sqrt(in1[gid].x*in1[gid].x + in1[gid].y*in1[gid].y
+ in2[gid].x*in2[gid].x + in2[gid].y*in2[gid].y);
}
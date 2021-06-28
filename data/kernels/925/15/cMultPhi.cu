#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void cMultPhi(double2* in1, double* in2, double2* out){
double2 result;
unsigned int gid = getGid3d3d();
result.x = cos(in2[gid])*in1[gid].x - in1[gid].y*sin(in2[gid]);
result.y = in1[gid].x*sin(in2[gid]) + in1[gid].y*cos(in2[gid]);
out[gid] = result;
}
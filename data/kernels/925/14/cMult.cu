#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void cMult(double2* in1, double2* in2, double2* out){
unsigned int gid = getGid3d3d();
double2 result;
double2 tin1 = in1[gid];
double2 tin2 = in2[gid];
result.x = (tin1.x*tin2.x - tin1.y*tin2.y);
result.y = (tin1.x*tin2.y + tin1.y*tin2.x);
out[gid] = result;
}
#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void vecMult(double2 *in, double *factor, double2 *out){
double2 result;
unsigned int gid = getGid3d3d();
result.x = in[gid].x * factor[gid];
result.y = in[gid].y * factor[gid];
out[gid] = result;
}
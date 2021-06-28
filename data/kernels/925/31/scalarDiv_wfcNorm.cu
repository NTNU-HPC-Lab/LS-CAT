#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void scalarDiv_wfcNorm(double2* in, double dr, double* pSum, double2* out){
unsigned int gid = getGid3d3d();
double2 result;
double norm = sqrt((pSum[0])*dr);
result.x = (in[gid].x/norm);
result.y = (in[gid].y/norm);
out[gid] = result;
}
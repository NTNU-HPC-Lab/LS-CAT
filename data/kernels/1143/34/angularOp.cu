#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void angularOp(double omega, double dt, double2* wfc, double* xpyypx, double2* out){
unsigned int gid = getGid3d3d();
double2 result;
double op;
op = exp( -omega*xpyypx[gid]*dt);
result.x=wfc[gid].x*op;
result.y=wfc[gid].y*op;
out[gid]=result;
}
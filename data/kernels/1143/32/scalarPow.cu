#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__device__ double2 pow(double2 a, int b){
double r = sqrt(a.x*a.x + a.y*a.y);
double theta = atan(a.y / a.x);
return{pow(r,b)*cos(b*theta),pow(r,b)*sin(b*theta)};
}
__global__ void scalarPow(double2* in, double param, double2* out){
unsigned int gid = getGid3d3d();
double2 result;
result.x = pow(in[gid].x, param);
result.y = pow(in[gid].y, param);
out[gid] = result;
}
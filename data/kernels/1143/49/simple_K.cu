#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void simple_K(double *xp, double *yp, double *zp, double mass, double *K){

unsigned int gid = getGid3d3d();
unsigned int xid = blockDim.x*blockIdx.x + threadIdx.x;
unsigned int yid = blockDim.y*blockIdx.y + threadIdx.y;
unsigned int zid = blockDim.z*blockIdx.z + threadIdx.z;
K[gid] = (HBAR*HBAR/(2*mass))*(xp[xid]*xp[xid] + yp[yid]*yp[yid]
+ zp[zid]*zp[zid]);
}
#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void kring_rotation_Az(double *x, double *y, double *z, double xMax, double yMax, double zMax, double omegaX, double omegaY, double omegaZ, double omega, double fudge, double *A){
int gid = getGid3d3d();
int xid = blockDim.x*blockIdx.x + threadIdx.x;
int yid = blockDim.y*blockIdx.y + threadIdx.y;
double r = sqrt(x[xid]*x[xid] + y[yid]*y[yid]);
A[gid] = r*omega*omegaX;
}
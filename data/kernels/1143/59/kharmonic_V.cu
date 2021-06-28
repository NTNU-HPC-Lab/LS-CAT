#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void kharmonic_V(double *x, double *y, double *z, double* items, double *Ax, double *Ay, double *Az, double *V){

int gid = getGid3d3d();
int xid = blockDim.x*blockIdx.x + threadIdx.x;
int yid = blockDim.y*blockIdx.y + threadIdx.y;
int zid = blockDim.z*blockIdx.z + threadIdx.z;

double V_x = items[3]*(x[xid]+items[6]);
double V_y = items[10]*items[4]*(y[yid]+items[7]);
double V_z = items[11]*items[5]*(z[zid]+items[8]);

V[gid] = 0.5*items[9]*((V_x*V_x + V_y*V_y + V_z*V_z)
+ (Ax[gid]*Ax[gid] + Ay[gid]*Ay[gid] + Az[gid]*Az[gid]));
}
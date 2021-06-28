#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void ktorus_V(double *x, double *y, double *z, double* items, double *Ax, double *Ay, double *Az, double *V){

int gid = getGid3d3d();
int xid = blockDim.x*blockIdx.x + threadIdx.x;
int yid = blockDim.y*blockIdx.y + threadIdx.y;
int zid = blockDim.z*blockIdx.z + threadIdx.z;

double rad = sqrt((x[xid] - items[6]) * (x[xid] - items[6])
+ (y[yid] - items[7]) * (y[yid] - items[7]))
- 0.5*items[0];
double omegaR = (items[3]*items[3] + items[4]*items[4]);
double V_tot = (2*items[5]*items[5]*(z[zid] - items[8])*(z[zid] - items[8])
+ omegaR*(rad*rad + items[12]*rad*z[zid]));
V[gid] = 0.5*items[9]*(V_tot
+ Ax[gid]*Ax[gid]
+ Ay[gid]*Ay[gid]
+ Az[gid]*Az[gid]);
}
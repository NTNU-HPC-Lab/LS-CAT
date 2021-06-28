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
__global__ void ktorus_wfc(double *x, double *y, double *z, double *items, double winding, double *phi, double2 *wfc){

int gid = getGid3d3d();
int xid = blockDim.x*blockIdx.x + threadIdx.x;
int yid = blockDim.y*blockIdx.y + threadIdx.y;
int zid = blockDim.z*blockIdx.z + threadIdx.z;

double rad = sqrt((x[xid] - items[6]) * (x[xid] - items[6])
+ (y[yid] - items[7]) * (y[yid] - items[7]))
- 0.5*items[0];

wfc[gid].x = exp(-( pow((rad)/(items[14]*items[15]*0.5),2) +
pow((z[zid])/(items[14]*items[17]*0.5),2) ) );
wfc[gid].y = 0.0;
}
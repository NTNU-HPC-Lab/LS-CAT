#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void kstd_wfc(double *x, double *y, double *z, double *items, double winding, double *phi, double2 *wfc){

int gid = getGid3d3d();
int xid = blockDim.x*blockIdx.x + threadIdx.x;
int yid = blockDim.y*blockIdx.y + threadIdx.y;
int zid = blockDim.z*blockIdx.z + threadIdx.z;

phi[gid] = -fmod(winding*atan2(y[yid], x[xid]),2*PI);

wfc[gid].x = exp(-(x[xid]*x[xid]/(items[14]*items[14]*items[15]*items[15])
+ y[yid]*y[yid]/(items[14]*items[14]*items[16]*items[16])
+ z[zid]*z[zid]/(items[14]*items[14]*items[17]*items[17])))
* cos(phi[gid]);
wfc[gid].y = -exp(-(x[xid]*x[xid]/(items[14]*items[14]*items[15]*items[15])
+ y[yid]*y[yid]/(items[14]*items[14]*items[16]*items[16])
+ z[zid]*z[zid]/(items[14]*items[14]*items[17]*items[17])))
* sin(phi[gid]);

}
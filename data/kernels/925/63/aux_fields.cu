#include "includes.h"
__device__ unsigned int getGid3d3d(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
+ gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
+ (threadIdx.y * blockDim.x)
+ (threadIdx.z * (blockDim.x * blockDim.y)) + threadIdx.x;
return threadId;
}
__global__ void aux_fields(double *V, double *K, double gdt, double dt, double* Ax, double *Ay, double* Az, double *px, double *py, double *pz, double* pAx, double* pAy, double* pAz, double2* GV, double2* EV, double2* GK, double2* EK, double2* GpAx, double2* GpAy, double2* GpAz, double2* EpAx, double2* EpAy, double2* EpAz){
int gid = getGid3d3d();
int xid = blockDim.x*blockIdx.x + threadIdx.x;
int yid = blockDim.y*blockIdx.y + threadIdx.y;
int zid = blockDim.z*blockIdx.z + threadIdx.z;

GV[gid].x = exp(-V[gid]*(gdt/(2*HBAR)));
GK[gid].x = exp(-K[gid]*(gdt/HBAR));
GV[gid].y = 0.0;
GK[gid].y = 0.0;

// Ax and Ay will be calculated here but are used only for
// debugging. They may be needed later for magnetic field calc

pAx[gid] = Ax[gid] * px[xid];
pAy[gid] = Ay[gid] * py[yid];
pAz[gid] = Az[gid] * pz[zid];

GpAx[gid].x = exp(-pAx[gid]*gdt);
GpAx[gid].y = 0;
GpAy[gid].x = exp(-pAy[gid]*gdt);
GpAy[gid].y = 0;
GpAz[gid].x = exp(-pAz[gid]*gdt);
GpAz[gid].y = 0;

EV[gid].x=cos(-V[gid]*(dt/(2*HBAR)));
EV[gid].y=sin(-V[gid]*(dt/(2*HBAR)));
EK[gid].x=cos(-K[gid]*(dt/HBAR));
EK[gid].y=sin(-K[gid]*(dt/HBAR));

EpAz[gid].x=cos(-pAz[gid]*dt);
EpAz[gid].y=sin(-pAz[gid]*dt);
EpAy[gid].x=cos(-pAy[gid]*dt);
EpAy[gid].y=sin(-pAy[gid]*dt);
EpAx[gid].x=cos(-pAx[gid]*dt);
EpAx[gid].y=sin(-pAx[gid]*dt);
}
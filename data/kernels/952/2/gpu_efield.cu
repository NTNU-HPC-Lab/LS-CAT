#include "includes.h"
__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y)
{
return NX*y+x;
}
__global__ void gpu_efield(double *fi, double *ex, double *ey){

unsigned int y = blockIdx.y;
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int xp1 = (x + 1) % NX;
unsigned int yp1 = (y + 1) % NY;
unsigned int xm1 = (NX + x - 1) % NX;
unsigned int ym1 = (NY + y - 1) % NY;
double phi  = fi[gpu_scalar_index(x, y)];
double phiL = fi[gpu_scalar_index(xm1, y)];
double phiR = fi[gpu_scalar_index(xp1, y)];
double phiU = fi[gpu_scalar_index(x, yp1)];
double phiD = fi[gpu_scalar_index(x, ym1)];
ex[gpu_scalar_index(x, y)] = 0.5*(phiL - phiR) / dx;
ey[gpu_scalar_index(x, y)] = 0.5*(phiD - phiU) / dy;
}
#include "includes.h"
__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y, unsigned int z)
{
return NX*(NY*z + y)+x;
}
__global__ void gpu_efield(double *fi, double *ex, double *ey, double *ez){

unsigned int y = blockIdx.y;
unsigned int z = blockIdx.z;
unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int xp1 = (x + 1) % NX;
unsigned int yp1 = (y + 1) % NY;
unsigned int zp1 = (z + 1) % NZ;

unsigned int xm1 = (NX + x - 1) % NX;
unsigned int ym1 = (NY + y - 1) % NY;
unsigned int zm1 = (NZ + z - 1) % NZ;

ex[gpu_scalar_index(x, y, z)] = 0.5*(fi[gpu_scalar_index(xm1,y,z)] - fi[gpu_scalar_index(xp1, y, z)]) / dx;
ey[gpu_scalar_index(x, y, z)] = 0.5*(fi[gpu_scalar_index(x, ym1, z)] - fi[gpu_scalar_index(x, yp1, z)]) / dy;
ez[gpu_scalar_index(x, y, z)] = 0.5*(fi[gpu_scalar_index(x, y, zm1)] - fi[gpu_scalar_index(x, y, zp1)]) / dz;
}
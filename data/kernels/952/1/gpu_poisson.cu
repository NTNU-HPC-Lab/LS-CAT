#include "includes.h"
__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y)
{
return NX*y+x;
}
__device__ __forceinline__ size_t gpu_s_scalar_index(unsigned int x, unsigned int y)
{
return (2*RAD + nThreads)*y + x;
}
__global__ void gpu_poisson(double *c, double *fi,double *R){
unsigned int y   = blockIdx.y;
unsigned int x   = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int s_y = threadIdx.y + RAD;
unsigned int s_x = threadIdx.x + RAD;
unsigned int xp1 = (x + blockDim.x) % NX;
unsigned int yp1 = (y + blockDim.y) % NY;
unsigned int xm1 = (NX + x - 1) % NX;
unsigned int ym1 = (NY + y - 1) % NY;
__shared__ double s_in[(2*RAD + nThreads)*3];
// load to shared memory (regular cells)
s_in[gpu_s_scalar_index(s_x,s_y)] = fi[gpu_scalar_index(x, y)];

// load halo cells
if (threadIdx.x < RAD) {
s_in[gpu_s_scalar_index(s_x - RAD, s_y)] = fi[gpu_scalar_index(xm1, y)];
s_in[gpu_s_scalar_index(s_x + blockDim.x, s_y)] = fi[gpu_scalar_index(xp1, y)];
}
if (threadIdx.y < RAD) {
s_in[gpu_s_scalar_index(s_x, s_y - RAD)] = fi[gpu_scalar_index(x, ym1)];
s_in[gpu_s_scalar_index(s_x, s_y + blockDim.y)] = fi[gpu_scalar_index(x, yp1)];
}
// Boundary conditions
if (y == 0) {
fi[gpu_scalar_index(x, y)] = voltage;
return;
}
if (y == NY - 1) {
fi[gpu_scalar_index(x, y)] = 0.0;
return;
}
__syncthreads();

double charge    = c[gpu_scalar_index(x, y)];
//double phi       = fi[gpu_scalar_index(x, y)];
//double phiL      = fi[gpu_scalar_index(xm1, y)];
//double phiR      = fi[gpu_scalar_index(xp1, y)];
//double phiU      = fi[gpu_scalar_index(x, yp1)];
//double phiD      = fi[gpu_scalar_index(x, ym1)];

double phi  = s_in[gpu_s_scalar_index(s_x, s_y)];
double phiL = s_in[gpu_s_scalar_index(s_x-1, s_y)];
double phiR = s_in[gpu_s_scalar_index(s_x+1, s_y)];
double phiU = s_in[gpu_s_scalar_index(s_x, s_y+1)];
double phiD = s_in[gpu_s_scalar_index(s_x, s_y-1)];

double source    = (charge / eps) * dx *dx; // Right hand side of the equation
double phi_old   = phi;
phi = 0.25 * (phiL + phiR + phiU + phiD + source);
// Record the error
R[gpu_scalar_index(x, y)] = fabs(phi - phi_old);

//__syncthreads();
fi[gpu_scalar_index(x, y)] = phi;
//if (x == 5 && y == 5) printf("%g\n", phi);
}
#include "includes.h"
__global__ void jacobi_init( const int x_inner, const int y_inner, const int halo_depth, const double* density, const double* energy, const double rx, const double ry, double* kx, double* ky, double* u0, double* u, const int coefficient)
{
const int gid = threadIdx.x+blockIdx.x*blockDim.x;
if(gid >= x_inner*y_inner) return;

const int x = x_inner + 2*halo_depth;
const int col = gid % x_inner;
const int row = gid / x_inner;
const int off0 = halo_depth*(x + 1);
const int index = off0 + col + row*x;

const double u_temp = energy[index]*density[index];
u0[index] = u_temp;
u[index] = u_temp;

if(row == 0 || col == 0) return;

double density_center;
double density_left;
double density_down;

if(coefficient == CONDUCTIVITY)
{
density_center = density[index];
density_left = density[index-1];
density_down = density[index-x];
}
else if(coefficient == RECIP_CONDUCTIVITY)
{
density_center = 1.0/density[index];
density_left = 1.0/density[index-1];
density_down = 1.0/density[index-x];
}

kx[index] = rx*(density_left+density_center) /
(2.0*density_left*density_center);
ky[index] = ry*(density_down+density_center) /
(2.0*density_down*density_center);
}
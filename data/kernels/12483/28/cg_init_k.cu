#include "includes.h"
__global__ void cg_init_k( const int x_inner, const int y_inner, const int halo_depth, const double* w, double* kx, double* ky, double rx, double ry)
{
const int gid = threadIdx.x+blockIdx.x*blockDim.x;
if(gid >= x_inner*y_inner) return;

const int x = x_inner + 2*halo_depth-1;
const int col = gid % x_inner;
const int row = gid / x_inner;
const int off0 = halo_depth*(x + 1);
const int index = off0 + col + row*x;

kx[index] = rx*(w[index-1]+w[index]) /
(2.0*w[index-1]*w[index]);
ky[index] = ry*(w[index-x]+w[index]) /
(2.0*w[index-x]*w[index]);
}
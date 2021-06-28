#include "includes.h"
__global__ void cg_calc_p( const int x_inner, const int y_inner, const int halo_depth, const double beta, const double* r, double* p)
{
const int gid = threadIdx.x+blockIdx.x*blockDim.x;
if(gid >= x_inner*y_inner) return;

const int x = x_inner + 2*halo_depth;
const int col = gid % x_inner;
const int row = gid / x_inner;
const int off0 = halo_depth*(x + 1);
const int index = off0 + col + row*x;

p[index] = r[index] + beta*p[index];
}
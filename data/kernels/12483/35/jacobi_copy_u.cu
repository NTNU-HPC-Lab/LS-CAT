#include "includes.h"
__global__ void jacobi_copy_u( const int x_inner, const int y_inner, const double* src, double* dest)
{
const int gid = threadIdx.x+blockIdx.x*blockDim.x;

if(gid < x_inner*y_inner)
{
dest[gid] = src[gid];
}
}
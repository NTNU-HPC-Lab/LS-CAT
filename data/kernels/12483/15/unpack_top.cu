#include "includes.h"
__global__ void unpack_top( const int x, const int y, const int halo_depth, double* field, double* buffer, const int depth)
{
const int x_inner = x - 2*halo_depth;

const int gid = threadIdx.x+blockDim.x*blockIdx.x;
if(gid >= x_inner*depth) return;

const int lines = gid / x_inner;
const int offset = x*(y - halo_depth) + lines*2*halo_depth;
field[offset+gid] = buffer[gid];
}
#include "includes.h"
__global__ void unpack_right( const int x, const int y, const int halo_depth, double* field, double* buffer, const int depth)
{
const int y_inner = y - 2*halo_depth;

const int gid = threadIdx.x+blockDim.x*blockIdx.x;
if(gid >= y_inner*depth) return;

const int lines = gid / depth;
const int offset = x - halo_depth + lines*(x - depth);
field[offset+gid] = buffer[gid];
}
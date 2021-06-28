#include "includes.h"
__global__ void set_chunk_data_vertices( int x, int y, int halo_depth, double dx, double dy, double x_min, double y_min, double* vertex_x, double* vertex_y, double* vertex_dx, double* vertex_dy)
{
const int gid = blockIdx.x*blockDim.x+threadIdx.x;

if(gid < x+1)
{
vertex_x[gid] = x_min + dx*(gid-halo_depth);
vertex_dx[gid] = dx;
}

if(gid < y+1)
{
vertex_y[gid] = y_min + dy*(gid-halo_depth);
vertex_dy[gid] = dy;
}
}
#include "includes.h"
__global__ void set_chunk_data( int x, int y, double dx, double dy, double* cell_x, double* cell_y, double* cell_dx, double* cell_dy, double* vertex_x, double* vertex_y, double* volume, double* x_area, double* y_area)
{
const int gid = blockIdx.x*blockDim.x+threadIdx.x;

if(gid < x)
{
cell_x[gid] = 0.5*(vertex_x[gid]+vertex_x[gid+1]);
cell_dx[gid] = dx;
}

if(gid < y)
{
cell_y[gid] = 0.5*(vertex_y[gid]+vertex_y[gid+1]);
cell_dy[gid] = dy;
}

if(gid < x*y)
{
volume[gid] = dx*dy;
}

if(gid < (x+1)*y)
{
x_area[gid] = dy;
}

if(gid < x*(y+1))
{
y_area[gid] = dx;
}
}
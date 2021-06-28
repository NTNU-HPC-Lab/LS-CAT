#include "includes.h"

#define IDX2D(a, i, stride, j) ((a)[(i)*(stride) + (j)])

__global__ void sim_kernel_tiled(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
extern __shared__ double z_tile[];

const int block_mesh_x = blockDim.x*blockIdx.x + 1;
const int block_mesh_y = blockDim.y*blockIdx.y + 1;

const int mesh_xx = block_mesh_x + threadIdx.x;
const int mesh_xy = block_mesh_y + threadIdx.y;

// We have to read into the tile BEFORE dropping threads so that it's actually fully
// initialized!
const double z_val = IDX2D(z_tile, threadIdx.y, blockDim.x, threadIdx.x)
= IDX2D(z, mesh_xy, nx, mesh_xx);

if (mesh_xx >= nx-1 || mesh_xy >= ny-1)
return;

__syncthreads();

double ax, ay;
if (1 <= threadIdx.x && threadIdx.x <= blockDim.x-2)
ax = dx2inv*(IDX2D(z_tile, threadIdx.y, blockDim.x, threadIdx.x-1)
+ IDX2D(z_tile, threadIdx.y, blockDim.x, threadIdx.x+1)
- 2.0*z_val);
else {
const int n = threadIdx.x == 0 ? -1 : +1;
ax = dx2inv*(IDX2D(z, mesh_xy, nx, mesh_xx+n)
+ IDX2D(z_tile, threadIdx.y, blockDim.x, threadIdx.x-n)
- 2.0*z_val);
}

if (1 <= threadIdx.y && threadIdx.y <= blockDim.y-2)
ay = dy2inv*(IDX2D(z_tile, threadIdx.y-1, blockDim.x, threadIdx.x)
+ IDX2D(z_tile, threadIdx.y+1, blockDim.x, threadIdx.x)
- 2.0*z_val);
else {
const int n = threadIdx.y == 0 ? -1 : +1;
ay = dx2inv*(IDX2D(z, mesh_xy+n, nx, mesh_xx)
+ IDX2D(z_tile, threadIdx.y-n, blockDim.x, threadIdx.x)
- 2.0*z_val);
}

const double v_val = IDX2D(v, mesh_xy, nx, mesh_xx) += (ax+ay)/2.0*dt;
IDX2D(z, mesh_xy, nx, mesh_xx) += dt*v_val;
}
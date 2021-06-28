#include "includes.h"

#define IDX2D(a, i, stride, j) ((a)[(i)*(stride) + (j)])

__device__ inline double warp_accel_shfl(double z, double d_2inv, int shfl_mask) {
return d_2inv*(__shfl_down_sync(shfl_mask, z, 1) + __shfl_up_sync(shfl_mask, z, 1) - 2.0*z);
}
__global__ void sim_kernel_shfl(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
__shared__ double z_tile[WARP_SIZE][WARP_SIZE];
__shared__ double ay_tile[WARP_SIZE][WARP_SIZE];

const int block_mesh_x = warpSize*blockIdx.x + 1;
const int block_mesh_y = warpSize*blockIdx.y + 1;

const int mesh_xx = block_mesh_x + threadIdx.x;
const int mesh_xy = block_mesh_y + threadIdx.y;
const int mesh_yx = block_mesh_x + threadIdx.y;
const int mesh_yy = block_mesh_y + threadIdx.x;

const double z_val_x = z_tile[threadIdx.y][threadIdx.x] = IDX2D(z, mesh_xy, nx, mesh_xx);

if (mesh_xx >= nx-1 || mesh_xy >= ny-1 /*|| mesh_yx > nx-1 || mesh_yy >= ny-1*/)
return;

__syncthreads();

const double z_val_y = z_tile[threadIdx.x][threadIdx.y];

//    const int shfl_mask = 0x7 << (threadIdx.x - 1);
const int shfl_mask = 0x7 << (threadIdx.x - 1);

double ax = warp_accel_shfl(z_val_x, dx2inv, shfl_mask);
double ay = warp_accel_shfl(z_val_y, dy2inv, shfl_mask);
if (threadIdx.x == 0 || threadIdx.x == warpSize-1) {
const int n = threadIdx.x == 0 ? -1 : +1;
ax = dx2inv*(IDX2D(z, mesh_xy, nx, mesh_xx+n) + z_tile[threadIdx.y][threadIdx.x-n]
- 2.0*z_val_x);
ay = dy2inv*(IDX2D(z, mesh_yy+n, nx, mesh_yx) + z_tile[threadIdx.x-n][threadIdx.y]
- 2.0*z_val_y);
}

ay_tile[threadIdx.x][threadIdx.x] = ay;
__syncthreads();
ay = ay_tile[threadIdx.y][threadIdx.x];

const double v_val = (IDX2D(v, mesh_xy, nx, mesh_xx) += (ax+ay)/2.0*dt);
IDX2D(z, mesh_xy, nx, mesh_xx) += dt*v_val;
}
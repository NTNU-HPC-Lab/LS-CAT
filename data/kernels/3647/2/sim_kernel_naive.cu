#include "includes.h"

#define IDX2D(a, i, stride, j) ((a)[(i)*(stride) + (j)])

__global__ void sim_kernel_naive(double *z, double *v, size_t nx, size_t ny, double dx2inv, double dy2inv, double dt) {
const int mesh_x = blockIdx.x*blockDim.x + threadIdx.x + 1;
const int mesh_y = blockIdx.y*blockDim.y + threadIdx.y + 1;

if (mesh_x >= nx-1 || mesh_y >= ny-1) return;

const double z_val = IDX2D(z, mesh_y, nx, mesh_x);
const double ax = dx2inv*(IDX2D(z, mesh_y, nx, mesh_x-1) + IDX2D(z, mesh_y, nx, mesh_x+1)
- 2.0*z_val);
const double ay = dy2inv*(IDX2D(z, mesh_y-1, nx, mesh_x) + IDX2D(z, mesh_y+1, nx, mesh_x)
- 2.0*z_val);

double const v_val = IDX2D(v, mesh_y, nx, mesh_x) + dt*(ax + ay)/2;
IDX2D(v, mesh_y, nx, mesh_x) = v_val;
IDX2D(z, mesh_y, nx, mesh_x) += dt*v_val;
}
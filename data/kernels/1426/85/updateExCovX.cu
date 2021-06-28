#include "includes.h"
__global__ void updateExCovX(double *e_x_cov_x, double gauss_d2, int valid_voxel_num)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

for (int i = id; i < valid_voxel_num; i += stride) {
e_x_cov_x[i] *= gauss_d2;
}
}
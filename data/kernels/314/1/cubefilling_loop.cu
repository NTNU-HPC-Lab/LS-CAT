#include "includes.h"
__global__ void cubefilling_loop(const float* image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size, int scale_xy, int scale_eps, dim3 dimensions_down)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
if (i < dimensions_down.x && j < dimensions_down.y) {

size_t cube_idx_1 = i + dimensions_down.x*j;
#pragma unroll
for (int ii = 0; ii < scale_xy; ii++)
{
#pragma unroll
for (int jj = 0; jj < scale_xy; jj++)
{
size_t i_idx = scale_xy*i + ii;
size_t j_idx = scale_xy*j + jj;
if (i_idx < image_size.x && j_idx < image_size.y)
{

float k = image[i_idx + image_size.x*j_idx];
size_t cube_idx_2 = cube_idx_1 + dimensions_down.x*dimensions_down.y*floorf(k / scale_eps);
dev_cube_wi[cube_idx_2] += k;
dev_cube_w[cube_idx_2] += 1.0f;
}

}
}
}


}
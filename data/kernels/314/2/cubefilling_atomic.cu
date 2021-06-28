#include "includes.h"
__global__ void cubefilling_atomic(const float* image, float *dev_cube_wi, float *dev_cube_w, const dim3 image_size, int scale_xy, int scale_eps, dim3 dimensions_down)
{
const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
if (i < image_size.x && j < image_size.y) {
const float k = image[i + image_size.x*j];
const size_t cube_idx = (i / scale_xy) + dimensions_down.x*(j / scale_xy) + dimensions_down.x*dimensions_down.y*((int)k / scale_eps);


atomicAdd(&dev_cube_wi[cube_idx], k);
atomicAdd(&dev_cube_w[cube_idx], 1.0f);

}


}
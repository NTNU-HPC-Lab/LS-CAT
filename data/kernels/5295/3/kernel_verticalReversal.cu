#include "includes.h"




__global__ void kernel_verticalReversal(dev_t *src, dev_t *dst, uint pitch_src, uint pitch_dst, uint pixel_w, uint pixel_h)
{
unsigned int dim_x = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int dim_y = blockDim.y * blockIdx.y + threadIdx.y;

if (dim_x < pixel_w && dim_y < pixel_h) {
unsigned int rgba = *((uint32_t*)src + dim_y * pitch_src / 4 + dim_x);
*((uint32_t*)dst + (pixel_h - dim_y) * pitch_dst / 4 + dim_x) = rgba;
}
}
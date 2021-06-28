#include "includes.h"




__device__ u_char clamp(float t)
{
if (t < 0) {
return 0;
} else if (t > 255){
return 255;
}

return t;
}
__global__ void kernel_colorSpaceRGBAToYUV420P(dev_t *src, dev_t *dst, int pitch_src, int pitch_dst, int w, int h)
{
unsigned int dim_x = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int dim_y = blockDim.y * blockIdx.y + threadIdx.y;

int r,g,b;
unsigned int rgba;
if (dim_x < w && dim_y < h) {
rgba = *((uint32_t*)dst + dim_y * pitch_dst / 4 + dim_x);
r = (rgba >> 24);
g = (rgba >> 16) & 0xff;
b = (rgba >> 8) & 0xff;
*((u_char*)src + dim_y * pitch_src + dim_x) = clamp(0.299 * r + 0.587 * g + 0.114 * b);
*((u_char*)src + (h + dim_y / 4) * pitch_src + dim_x / 2) = clamp(-0.1687 * r - 0.3313 * g + 0.5 * b + 128);
*((u_char*)src + (h * 5 + dim_y) / 4 * pitch_src + dim_x / 2) = clamp(0.5 * r - 0.4187 * g - 0.0813 * b + 128);
}
}
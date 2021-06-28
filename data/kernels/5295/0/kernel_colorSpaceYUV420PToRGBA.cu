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
__global__ void kernel_colorSpaceYUV420PToRGBA(dev_t *src, dev_t *dst, int pitch_src, int pitch_dst, int w, int h)
{
unsigned int dim_x = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int dim_y = blockDim.y * blockIdx.y + threadIdx.y;

int r,g,b,y,u,v;
if (dim_x < w && dim_y < h) {
y = *((u_char*)src + dim_y * pitch_src + dim_x);
u = *((u_char*)src + (h + dim_y / 4) * pitch_src + dim_x / 2);
v = *((u_char*)src + (h * 5 + dim_y) / 4 * pitch_src + dim_x / 2);
r = clamp(y + 1.402 * (v - 128) + 0.5);
g = clamp(y - 0.34414 * (u - 128) - 0.71414 * (v - 128) + 0.5);
b = clamp(y + 1.772 * (u - 128) + 0.5);
//        *((uint32_t*)dst + dim_y * pitch_dst / 4 + dim_x) = (r << 24) + (g << 16) + (b << 8);
*((u_char*)dst + dim_y * pitch_dst + dim_x * 4) = r;
*((u_char*)dst + dim_y * pitch_dst + dim_x * 4 + 1) = g;
*((u_char*)dst + dim_y * pitch_dst + dim_x * 4 + 2) = b;
*((u_char*)dst + dim_y * pitch_dst + dim_x * 4 + 3) = 255;
}
}
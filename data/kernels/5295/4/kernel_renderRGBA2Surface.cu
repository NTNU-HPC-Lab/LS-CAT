#include "includes.h"




__global__ void kernel_renderRGBA2Surface(cudaSurfaceObject_t surface, dev_t *src, int pitch, int pixel_w, int pixel_h)
{
unsigned int dim_x = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int dim_y = blockDim.y * blockIdx.y + threadIdx.y;

if (dim_x < pixel_w && dim_y < pixel_h)
{
u_char r,g,b;
r = *((u_char*)src + dim_y * pitch + dim_x * 4);
g = *((u_char*)src + dim_y * pitch + dim_x * 4 + 1);
b = *((u_char*)src + dim_y * pitch + dim_x * 4 + 2);
uchar4 data = make_uchar4(r, g, b, 0xff);
surf2Dwrite(data, surface, dim_x * sizeof(uchar4), dim_y);
}
}
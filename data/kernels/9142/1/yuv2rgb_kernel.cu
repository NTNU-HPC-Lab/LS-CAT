#include "includes.h"

__global__ void yuv2rgb_kernel(int img_size, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *img_y, unsigned char *img_u, unsigned char *img_v){

int i = threadIdx.x + blockDim.x*blockIdx.x;
unsigned char y, cb, cr;

if(i < img_size){

y  = img_y[i];
cb = img_u[i] - 128;
cr = img_v[i] - 128;

img_r[i] = ( y + 1.402 * cr);
img_g[i] = ( y - 0.344 * cb - 0.714 * cr);
img_b[i] = ( y + 1.772 * cb);

}
}
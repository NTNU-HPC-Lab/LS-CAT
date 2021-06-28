#include "includes.h"

__global__ void rgb2yuv_kernel(int img_size, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *img_y, unsigned char *img_u, unsigned char *img_v) {

int i = threadIdx.x + blockDim.x * blockIdx.x;

if(i < img_size){
int r, g, b;
r = img_r[i];
g = img_g[i];
b = img_b[i];

img_y[i] = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
img_u[i] = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
img_v[i] = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
}
}
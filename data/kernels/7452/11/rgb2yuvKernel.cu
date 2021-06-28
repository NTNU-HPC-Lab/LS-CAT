#include "includes.h"
__global__ void rgb2yuvKernel(int *imgr,int *imgg,int *imgb,int *imgy,int *imgcb,int *imgcr, int n) {

int r, g, b;
int y, cb, cr;

int index;
index = threadIdx.x + blockIdx.x * blockDim.x;

if (index < n){
r = imgr[index];
g = imgg[index];
b = imgb[index];

y  = (int)( 0.299*r + 0.587*g +  0.114*b);
cb = (int)(-0.147*r - 0.289*g +  0.436*b);
cr = (int)( 0.615*r - 0.515*g - 0.1*b);

imgy[index] = y;
imgcb[index] = cb;
imgcr[index] = cr;
}
}
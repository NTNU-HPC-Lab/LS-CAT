#include "includes.h"
__global__ void yuv2rgbKernel(int *imgy,int *imgcb,int *imgcr, int *imgr,int *imgg,int *imgb, int n) {

int r, g, b;
int y, cb, cr;

int index;
index = threadIdx.x + blockIdx.x * blockDim.x;

if (index < n){
y = imgy[index];
cb = imgcb[index];
cr = imgcr[index];

r  = (int)( 1*y + 0*cb +  1.14*cr);
g = (int)( 1*y - 0.396*cb -  0.581*cr);
b = (int)( 1*y + 2.029*cb +  0*cr);

imgr[index] = r;
imgg[index] = g;
imgb[index] = b;
}
}
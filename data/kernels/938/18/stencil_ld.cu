#include "includes.h"
__global__ void stencil_ld(unsigned *X, unsigned *out, int width, int height){
int kernel[3][3] = { 0, -1, 0,
-1, 5, -1,
0, -1, 0};
int cikti;
int x  = blockIdx.y*32*width + blockIdx.x*32 + threadIdx.y*width + threadIdx.x; //current pixel

//if(x/width<1 || x/width>height-1 || x%width == width-1 || x%width == 1) return; // kenar noktalarinda

cikti  =(kernel[0][0]*X[x-width-1]       +
kernel[0][1]*X[x-width]         +
kernel[0][2]*X[x-width+1]       +
kernel[1][0]*X[x-1]             +
kernel[1][1]*X[x]               +
kernel[1][2]*X[x+1]             +
kernel[2][0]*X[x+width-1]       +
kernel[2][1]*X[x+width-1]       +
kernel[2][2]*X[x+width-1]);


if(cikti < 0)        out[x] = 0;
else if(cikti > 255) out[x] = 255;
else                 out[x] = cikti;

}
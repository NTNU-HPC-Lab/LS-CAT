#include "includes.h"
__global__ void ConvolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, int imageW, int imageH, int filterR){
int k;
float sum=0;
int row=blockDim.y*blockIdx.y+threadIdx.y;
int col=blockDim.x*blockIdx.x+threadIdx.x;

for (k = -filterR; k <= filterR; k++) {

int d = col+ k;

if (d >= 0 && d < imageW) {
sum += d_Src[row * imageW + d] * d_Filter[filterR - k];
}

d_Dst[row * imageW + col] = sum;
}

}
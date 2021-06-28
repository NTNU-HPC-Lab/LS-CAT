#include "includes.h"
__global__ void ConvolutionColGPU(double *d_Dst, double *d_Src, double *d_Filter, int imageW, int imageH, int filterR){
int k;
double sum=0;
int row=blockDim.y*blockIdx.y+threadIdx.y;
int col=blockDim.x*blockIdx.x+threadIdx.x;

for (k = -filterR; k <= filterR; k++) {

int d = row+ k;

if (d >= 0 && d < imageW) {
sum += d_Src[col +imageW* d] * d_Filter[filterR - k];
}

d_Dst[row * imageW + col] = sum;
}
}
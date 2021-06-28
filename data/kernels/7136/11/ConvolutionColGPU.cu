#include "includes.h"
__global__ void ConvolutionColGPU(float *d_Dst,float *d_Src,float *d_Filter,int filterR){
int x =threadIdx.x;
int y =threadIdx.y;

float sum=0;

for (int k = -filterR; k <= filterR; k++) {
int d = y + k;

if (d >= 0 && d < blockDim.y) {
sum += d_Src[d * blockDim.x + x] * d_Filter[filterR - k];
}

d_Dst[y * blockDim.x + x] = sum;
}
}
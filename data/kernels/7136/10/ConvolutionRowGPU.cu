#include "includes.h"
__global__ void ConvolutionRowGPU(float *d_Dst,float *d_Src,float *d_Filter,int filterR){
int x =threadIdx.x;
int y =threadIdx.y;
int k;

float sum=0;
for (k = -filterR; k <= filterR; k++) {
int d = x + k;
if (d >= 0 && d < blockDim.x) {
sum += d_Src[y*blockDim.x+d] * d_Filter[filterR- k];
}

d_Dst[y*blockDim.x+x] = sum;
}

}
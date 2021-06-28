#include "includes.h"
__global__ void convolutionColumnGPU(double *h_Dst, double *h_Src, double *h_Filter, int imageW, int imageH, int filterR){
int k;
double sum = 0;
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;
for (k = -filterR; k <= filterR; k++) {
int d = iy + k;
if (d >= 0 && d < imageH) {
sum += h_Src[d * imageW + ix] * h_Filter[filterR - k];
}
h_Dst[iy * imageW + ix] = sum;
}
}
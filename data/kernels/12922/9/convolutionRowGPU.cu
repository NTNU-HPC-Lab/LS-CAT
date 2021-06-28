#include "includes.h"
__global__ void convolutionRowGPU(double *h_Dst, double *h_Src, double *h_Filter, int imageW, int imageH, int filterR){

int k;
double sum = 0;
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;
for (k = -filterR; k <= filterR; k++) {
int d = ix + k;    //edw einai to pou tha paei to filtro gi auto elegxei apo katw kai an to d einia ektos oriwn eikonas
if (d >= 0 && d < imageW) {
sum += h_Src[iy * imageW + d] * h_Filter[filterR - k];
}
h_Dst[iy * imageW + ix] = sum;
}
}
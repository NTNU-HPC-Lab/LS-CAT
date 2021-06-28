#include "includes.h"
__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter, int imageW, int imageH, int filterR){
int k;
float sum=0;
int row=blockDim.y*blockIdx.y+threadIdx.y+filterR;
int col=blockDim.x*blockIdx.x+threadIdx.x+filterR;
int newImageW =imageW+filterR*2;
for (k = -filterR; k <= filterR; k++) {

int d = row+ k;

sum += d_Src[col +newImageW* d] * d_Filter[filterR - k];


}
d_Dst[row * newImageW + col] = sum;
}
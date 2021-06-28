#include "includes.h"
__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P,int width){
int Row = blockIdx.y*blockDim.y + threadIdx.y;
int Col = blockIdx.x*blockDim.x + threadIdx.x;

if ((Row < width)&&(Col < width)){
float Pvalue = 0;
for (int i = 0; i < width; ++i){
Pvalue += d_M[Row*width+i]*d_N[i*width+Col];
}
d_P[Row*width + Col] = Pvalue;
}
}
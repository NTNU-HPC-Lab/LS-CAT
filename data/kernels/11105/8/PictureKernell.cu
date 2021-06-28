#include "includes.h"
__global__ void PictureKernell(unsigned char* d_Pin, unsigned char* d_Pout, int n, int m) {
int Row = blockIdx.y*blockDim.y + threadIdx.y;
int Col = blockIdx.x*blockDim.x + threadIdx.x;

if((Row < m) && (Col < n)) {
d_Pout[Row*n+Col] = 2*d_Pin[Row*n+Col];
}
}
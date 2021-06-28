#include "includes.h"
__global__ void convolution_rgb(unsigned char *N,float *M,unsigned char* g,std::size_t cols, std::size_t rows,std::size_t mask_size){

int paddingSize = (( mask_size-1 )/2)*3;
unsigned int paddedH = cols + 2 * paddingSize;
unsigned int paddedW = rows*3 + 2 * paddingSize;

int i = blockIdx.x * blockDim.x + threadIdx.x + paddingSize ;
int j = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;
if( (i >= paddingSize) && (i < paddedW-paddingSize) && (j >= paddingSize) && (j<paddedH-paddingSize)) {
unsigned int oPixelPos = (i - paddingSize ) * cols + (j -paddingSize);
g[oPixelPos] = 0;
int iterationK = 0;
for(int k = -paddingSize; k <= paddingSize; k=k+3){
int iterationL = 0;
for(int l = -paddingSize; l<=paddingSize; l=l+3){
unsigned int iPixelPos = (i+k)*paddedH+(j+l);
unsigned int filtrePos = iterationK*mask_size + iterationL;

g[oPixelPos] += N[iPixelPos] * M[filtrePos];
iterationL++;
}
iterationK++;
}
}
}
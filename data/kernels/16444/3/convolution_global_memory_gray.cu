#include "includes.h"
__global__ void convolution_global_memory_gray(unsigned char *N,float *M,unsigned char* g,std::size_t cols, std::size_t rows,std::size_t mask_size){
int paddingSize = ( mask_size-1 )/2;
unsigned int paddedH = cols + 2 * paddingSize;
unsigned int paddedW = rows + 2 * paddingSize;

int i = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;
int j = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;

if( (j >= paddingSize) && (j < paddedW-paddingSize) && (i >= paddingSize) && (i<paddedH-paddingSize)) {
unsigned int oPixelPos = (j - paddingSize ) * cols + (i -paddingSize);
for(int k = -paddingSize; k <= paddingSize; k++){
for(int l = -paddingSize; l<=paddingSize; l++){
unsigned int iPixelPos = (j+l)*cols+(i+k);
unsigned int coefPos = (k + paddingSize) * mask_size + (l+ paddingSize);
g[oPixelPos] += N[iPixelPos] * M[coefPos];
}
}
}
}
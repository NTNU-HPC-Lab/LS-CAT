#include "includes.h"
__global__ void matMul(unsigned char *image,unsigned char *resImage,int rows,int cols){
/* it will modify each pixel */
//int ti = blockIdx.y*blockDim.y+threadIdx.y;
int tj = blockIdx.x*blockDim.x+threadIdx.x;
if(tj < rows*cols){
int pos = tj*chanDepth;
resImage[pos+BLUE] = image[pos+BLUE]*2;
resImage[pos+GREEN] = image[pos+GREEN]*2;
resImage[pos+RED] = image[pos+RED]*2;
}
}
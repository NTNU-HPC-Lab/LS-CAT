#include "includes.h"
__global__ void imgGray(unsigned char * d_image, unsigned char* d_imagegray, int width, int height){

int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;

if ((width > col) && (height > row)){
d_imagegray[row*width+col]=d_image[(row*width+col)*3+2]*0.299+d_image[(row*width+col)*3+1]*0.587+d_image[(row*width+col)*3]*0.114;
}
}
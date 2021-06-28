#include "includes.h"
__global__ void gray(unsigned char *In, unsigned char *Out,int Row, int Col){
int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;

if((row < Col) && (col < Row)){
Out[row*Row+col] = In[(row*Row+col)*3+2]*0.299 + In[(row*Row+col)*3+1]*0.587+ In[(row*Row+col)*3]*0.114;
}
}
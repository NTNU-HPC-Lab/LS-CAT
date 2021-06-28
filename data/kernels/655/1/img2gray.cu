#include "includes.h"
__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
int row = blockIdx.y*blockDim.y+threadIdx.y;
int col = blockIdx.x*blockDim.x+threadIdx.x;

if((row < height) && (col < width)){
int pos = (row*width+col)*Channels;

imageOutput[row*width+col] = imageInput[pos+RED]*0.299 + imageInput[pos+GREEN]*0.587 + imageInput[pos+BLUE]*0.114;
}
}
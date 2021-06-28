#include "includes.h"
__global__ void gpu_grey_and_thresh(unsigned char* Pout, unsigned char* Pin, int width, int height){

int channels = 3;
unsigned char thresh = 157;
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;

// check if pixel within range
if (col < width && row < height){
int gOffset = row * width + col;
int rgbOffset = gOffset * channels;
unsigned char r = Pin[rgbOffset  ];
unsigned char g = Pin[rgbOffset+1];
unsigned char b = Pin[rgbOffset+2];
unsigned char gval = 0.21f*r + 0.71f*g + 0.07f*b;

if(gval > thresh){
Pout[gOffset] = 255;
}
else {
Pout[gOffset] = 0;
}
}
}
#include "includes.h"
__global__ void gpu_grey_and_blur(unsigned char* Pout, unsigned char* Pin, int width, int height){

int channels = 3;
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;

// check if pixel within range
if (col < width && row < height){
int gOffset = row * width + col;
int rgbOffset = gOffset * channels;
unsigned char r = Pin[rgbOffset  ];
unsigned char g = Pin[rgbOffset+1];
unsigned char b = Pin[rgbOffset+2];
Pout[gOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
}
__syncthreads();

unsigned char k_size = 1;
int pixVal = 0;
int pixels = 0;
if (col < width && row < height){
for(int blurRow = -k_size; blurRow < k_size+1; ++blurRow){
for(int blurCol = -k_size; blurCol < k_size+1; ++blurCol){
int curRow = row + blurRow;
int curCol = col + blurCol;

if (curRow > -1 && curRow < height && curCol > -1 && curCol < width){
pixVal += Pout[curRow * width + curCol];
pixels++;
}
}
}
}
__syncthreads();
if (col < width && row < height) {
Pout[row * width + col] = (unsigned char) (pixVal / pixels);
}
}
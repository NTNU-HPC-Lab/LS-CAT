#include "includes.h"
__global__ void imageBlurKernel(unsigned char* d_image, int h, int w)
{
int Row = blockIdx.y * blockDim.y + threadIdx.y;
int Col = blockIdx.x * blockDim.x + threadIdx.x;

int blurSize = 8;

Row = Row * blurSize;
Col = Col * blurSize;

int r, g, b;
int p_r = 0;
int p_g = 0;
int p_b = 0;
int i, j;

if((Row+blurSize < h) && (Col+blurSize < w)){
for(i = 0; i < blurSize; i++)
for(j = 0; j < blurSize; j++){
r = d_image[4 * w * (Row+j) + 4 * (Col+i) + 0];
g = d_image[4 * w * (Row+j) + 4 * (Col+i) + 1];
b = d_image[4 * w * (Row+j) + 4 * (Col+i) + 2];

p_r += r;
p_g += g;
p_b += b;
}

p_r = p_r / (blurSize * blurSize);
p_g = p_g / (blurSize * blurSize);
p_b = p_b / (blurSize * blurSize);

for(i = 0; i < blurSize; i++)
for(j = 0; j < blurSize; j++){
d_image[4 * w * (Row+j) + 4 * (Col+i) + 0] = p_r;
d_image[4 * w * (Row+j) + 4 * (Col+i) + 1] = p_g;
d_image[4 * w * (Row+j) + 4 * (Col+i) + 2] = p_b;
}
}
}
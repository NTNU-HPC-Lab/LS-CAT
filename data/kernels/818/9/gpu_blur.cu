#include "includes.h"
__global__ void gpu_blur(unsigned char* Pout, unsigned char* Pin, int width, int height){
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;
int k_size = 3;

if (col < width && row < height){
int pixVal = 0;
int pixels = 0;

for(int blurRow = -k_size; blurRow < k_size+1; blurRow++){
for(int blurCol = -k_size; blurCol < k_size+1; blurCol++){
int curRow = row + blurRow;
int curCol = col + blurCol;

if (curRow > -1 && curRow < height && curCol > -1 && curCol < width){
pixVal += Pin[curRow * width + curCol];
pixels++;
}
}
}

Pout[row * width + col] = (unsigned char) (pixVal / pixels);
}
}
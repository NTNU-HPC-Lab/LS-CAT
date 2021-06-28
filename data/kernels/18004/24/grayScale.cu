#include "includes.h"
__global__ void grayScale(unsigned char* imgInput, unsigned char* imgOutput, int Row, int Col) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if ((row < Col) && (col < Row)) {
imgOutput[row * Row + col] = imgInput[(row * Row + col) * 3 + 2] * 0.299 + imgInput[(row * Row + col) * 3 + 1] * 0.587 + imgInput[(row * Row + col) * 3] * 0.114;
}
}
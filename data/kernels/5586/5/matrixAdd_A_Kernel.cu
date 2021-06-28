#include "includes.h"
__global__ void matrixAdd_A_Kernel(float* A, float* B, float* C, size_t pitch, int width){

//compute indexes
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;

int rowWidthWithPad = pitch/sizeof(float);


if(row < width && col < width)
C[row * rowWidthWithPad  + col] = A[row * rowWidthWithPad  + col] + B[row * rowWidthWithPad  + col];

}
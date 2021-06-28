#include "includes.h"
__global__ void matrixVectorMultKernel(float* fltMatrix, float* vec, float* output, int rows, int columns){

int row = blockDim.x * blockIdx.x + threadIdx.x;

if(row < rows){
float sum = 0.0f;
for (int col = 0; col < columns; ++col) {
sum += fltMatrix[row * columns + col] + vec[col];
}

output[row] = sum;
}

}
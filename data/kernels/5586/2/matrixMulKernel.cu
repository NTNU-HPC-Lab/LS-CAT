#include "includes.h"
__global__ void matrixMulKernel(float* d_M, float* d_N, float* d_P, int width){

//compute row and column of the target element to compute
int row = blockDim.y * blockIdx.y + threadIdx.y;
int col = blockDim.x * blockIdx.x + threadIdx.x;

//check for safety if target element is within matrix dimensions
if(row < width && col < width){
//perform "dot product" line and column
float sum = 0.0f;
for (int k = 0; k < width; ++k) {
sum += d_M[row * width + k] * d_N[k * width + col];
}
//assign target element value
d_P[row * width + col] = sum;
}
}
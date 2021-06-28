#include "includes.h"
__global__ void matrixMulKernel(float *C, float *A, float *B, int width, int height){
int tx = blockIdx.x * blockDim.x + threadIdx.x;
int ty = blockIdx.y * blockDim.y + threadIdx.y;
if(tx >= width || ty >= height)
return;

float sum = 0;
for(int i=0; i<width; ++i){
sum += A[ty * width + i] * B[i * width + tx];
}

C[ty * width + tx] = sum;
}
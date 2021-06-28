#include "includes.h"
/*
Modified from
https://github.com/zhxfl/CUDA-CNN
*/



__global__ void matrixTransKernel(float *A, int rows, int cols) {
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;

if (j >= cols || i >= rows) return;
float tmp = A[i * cols + j];
A[i * cols + j] = A[j * cols + i];
A[j * cols + i] = tmp;
}
#include "includes.h"
/*
Modified from
https://github.com/zhxfl/CUDA-CNN
*/



__global__ void elementwiseMul(float *x, float *y, float *z, int rows, int cols) {
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;

if (j >= cols || i >= rows) return;
z[i * cols + j] = x[i * cols + j] * y[i * cols + j];
}
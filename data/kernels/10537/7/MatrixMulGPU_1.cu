#include "includes.h"
__global__ void MatrixMulGPU_1(float *c, const float *a, const float *b, unsigned int WA, unsigned int WB) {
float sum = 0;
//找出该线程所在的行和列
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

//线程Thread(row, col)负责计算C(row, col)
for (int i = 0; i < WB; ++i) {
sum += a[row * WA + i] * b[i * WB + col];
}

c[row * WB + col] = sum;
}
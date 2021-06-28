#include "includes.h"
__global__ void squareMatrixMulKernel(int *c, int *a, int *b, int arrayWidth)
{
float sum = 0;

//여기서 threadIdx.x와 y는 행렬의 인덱스와 같다. 예시) 2x2행렬일때 00 01 10 11

for (int i = 0; i < arrayWidth; ++i)
{
float Aelement = a[threadIdx.y * arrayWidth + i];
float Belement = b[i*arrayWidth + threadIdx.x];
sum += Aelement * Belement;
}
c[threadIdx.y * arrayWidth + threadIdx.x] = sum;
}
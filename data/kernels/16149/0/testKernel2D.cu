#include "includes.h"
/*
Autor: Munesh Singh
Date: 08 March 2010
Vector addition using cudaMallocPitch
*/

const int width = 567;
const int height = 985;


__global__ void testKernel2D(float* M, float* N, float* P, size_t pitch) {
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;
if (row < width && col < width) {
float* row_M = (float*)((char*)M + row * pitch);
float* row_N = (float*)((char*)N + row * pitch);
float* row_P = (float*)((char*)P + row * pitch);

row_P[col] = row_M[col] + row_N[col];
}
}
#include "includes.h"
__global__ void sgemm_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta)
{
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

float element_c = 0.f;
for (int e = 0; e < K; e++)
element_c += A[row * K + e] * B[e * K + col];

C[row * N + col] = alpha * element_c + beta * C[row * N + col];
}
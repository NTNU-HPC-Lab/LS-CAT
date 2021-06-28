#include "includes.h"
__global__ void ReferenceGemm_kernel( int M, int N, int K, float alpha, float const *A, int lda, float const *B, int ldb, float beta, float *C, int ldc) {

int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;

if (i < M && j < N) {
float accumulator = 0;

for (int k = 0; k < K; ++k) {
accumulator += A[i + k * lda] * B[k + j * ldb];
}

C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
}
}
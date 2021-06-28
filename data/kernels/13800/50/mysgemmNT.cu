#include "includes.h"
__global__ void mysgemmNT( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
float c = 0.0f;
int m = blockIdx.x * blockDim.x + threadIdx.x;
int n = blockIdx.y * blockDim.y + threadIdx.y;
for (int i = 0; i < k; ++i) {
float a = A[m + i * lda];
float b = B[n + i * ldb];
c += a * b;
}
C[m+n*ldc] = C[m+n*ldc] * beta + alpha * c;
}
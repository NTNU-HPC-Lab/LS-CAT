#include "includes.h"
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
for (int i=0; i<N; i++) {
C[i] = A[i] + B[i];
}
}
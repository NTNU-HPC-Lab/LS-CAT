#include "includes.h"
__global__ void shared4R1Ws10ops2RG1WG(float *A, float *B, float *C, const int N)
{
__shared__ float Smem[512];

int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N)
Smem[threadIdx.x] = A[i];
__syncthreads();

if (i < N) {
C[i] = A[i] + B[i] - A[i]*A[i] + 3*B[i] - 4*A[i]*B[i] + B[i]*B[i]*7- 8+Smem[(threadIdx.x+1)%512]+Smem[(threadIdx.x+2)%512]+Smem[(threadIdx.x+3)%512]+Smem[(threadIdx.x+4)%512];
}
}
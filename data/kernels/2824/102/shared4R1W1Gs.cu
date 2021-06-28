#include "includes.h"
__global__ void shared4R1W1Gs(float *A, float *B, float *C, const int N)
{
__shared__ float Smem[512];

int i = blockIdx.x * blockDim.x + threadIdx.x;

Smem[threadIdx.x] = i;
__syncthreads();

if (i < N) {
C[i] = Smem[(threadIdx.x+1)%512]+Smem[(threadIdx.x+2)%512]+Smem[(threadIdx.x+3)%512]+Smem[(threadIdx.x+4)%512];
}
}
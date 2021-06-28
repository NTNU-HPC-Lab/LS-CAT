#include "includes.h"
__global__ void shared1R8C1W8C1G(float *A, float *B, float *C, const int N)
{
// compilador é esperto e aproveita o valor de i, mas faz 1W, 2 R nas outras posições da Shared
__shared__ float Smem[512];

int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) {
Smem[((threadIdx.x+1)*8)%512] = i;
C[i] = Smem[(threadIdx.x*8)%512];
}
}
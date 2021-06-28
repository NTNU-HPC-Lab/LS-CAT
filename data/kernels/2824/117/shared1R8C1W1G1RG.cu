#include "includes.h"
__global__ void shared1R8C1W1G1RG(float *A, float *B, float *C, const int N)
{
// compilador é esperto e aproveita o valor de i, mas faz 1W, 2 R nas outras posições da Shared
__shared__ float Smem[512];

int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) {
Smem[(threadIdx.x+1)%512] = A[i];
C[i] = Smem[(threadIdx.x*8)%512];
}
/*if ( blockIdx.x ==  2 && threadIdx.x < 32 ) {
printf("th %d smem %d\n",threadIdx.x,(threadIdx.x*8)%512);
}*/
}
#include "includes.h"
__global__  void GPUMult(int *A, int *B, int *C, int WIDTH)
{
int sol=0;
int i;i = threadIdx.x;
int j; j= threadIdx.y;

if (i < WIDTH && j < WIDTH) {
for (int k = 0; k < WIDTH; k++)
{
sol += A[j * WIDTH + k] * B[k * WIDTH + i];
}
C[j * WIDTH + i] = sol;
}

}
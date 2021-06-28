#include "includes.h"
__global__ void copy(int *src, int *dest)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
if (idx >= WIDTH || idy >= HEIGHT) return;

dest[idy * WIDTH + idx] = src[idy * WIDTH + idx]; // Copio tal cual con los mismos indices facil... :)
}
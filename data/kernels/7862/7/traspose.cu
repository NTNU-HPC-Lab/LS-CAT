#include "includes.h"
__global__ void traspose(int *src, int *dest)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
if (idx >= WIDTH || idy >= HEIGHT) return;

dest[idx * HEIGHT + idy] = src[idy * WIDTH + idx]; // Cambio el valor de la matriz a la traspuesta
// con los índices de acceso a la matriz...
}
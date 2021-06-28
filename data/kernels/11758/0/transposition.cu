#include "includes.h"
#define _CRT_SECURE_NO_WARNINGS

#define BLOCK_DIM 16


__global__ void transposition(int* matrix, int* matrixOut, int length, int width)
{
__shared__ int tempMatrix[BLOCK_DIM][BLOCK_DIM];//ðàçäåëÿåìàÿ ïàìÿòü

int temp;

int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

if ((i < length) && (j < width))
{
temp = j * length + i;
tempMatrix[threadIdx.y][threadIdx.x] = matrix[temp];
}

__syncthreads();

i = blockIdx.y * blockDim.y + threadIdx.x;//èíäåêñ áëîêà, ðàçìåðíîñòü áëîêà, èíäåêñ ïîòîêà
j = blockIdx.x * blockDim.x + threadIdx.y;

if ((i < width) && (j < length))
{
temp = j * width + i;
matrixOut[temp] = tempMatrix[threadIdx.x][threadIdx.y];
}
}
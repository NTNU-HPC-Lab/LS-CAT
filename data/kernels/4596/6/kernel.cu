#include "includes.h"
__global__ void kernel(int* D, int* q, int k){

// Find index of i row and j column of the distance array
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

if(D[i * N + j] > D[i * N + k] + D[k * N + j])
{
D[i * N + j] = D[i * N + k] + D[k * N + j];
q[i * N + j] = k;
}
}
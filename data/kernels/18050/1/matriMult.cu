#include "includes.h"
__global__ void matriMult(int* m, int* n, int* p, int size){
// Calculate Row and Coulmn
int row = blockIdx.y * blockDim.y + threadIdx.y;
int column = blockIdx.x * blockDim.x + threadIdx.x;
int p_sum = 0;
for(int i = 0; i < size; i++){
p_sum += m[row * size + i] * n[i * size + column];
}

p[row * size + column] = p_sum;
}
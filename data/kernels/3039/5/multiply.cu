#include "includes.h"
__global__ void multiply(int* a, int* b, int* c, int x, int y) {
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

int temp =  0;
if(row < x && col < x) {
for(int i = 0; i < y; i++) {
temp += a[row * y + i] * b[i * x + col];
}
}
c[row * x + col] = temp;
}
#include "includes.h"
__global__ void set_zero(float *A, float *I, int n, int i){
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x < n && y < n){
if (x != i){
if (y == i){
A[x*n + y] = 0;
}
}
}
}
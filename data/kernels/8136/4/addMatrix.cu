#include "includes.h"
__global__ void addMatrix(int *c, int *a, int *b){
int j = blockIdx.x*blockDim.x + threadIdx.x;
int i = blockIdx.y*blockDim.y + threadIdx.y;
*(c + blockDim.y*i + j) = *(a + blockDim.y*i + j) + *(b + blockDim.y*i + j);
}
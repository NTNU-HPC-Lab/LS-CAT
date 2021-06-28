#include "includes.h"
__global__ void matmul_partition(const float *a, const float *b, float *c, int n){
const int TILE_WIDTH = 8;
__shared__ float na[TILE_WIDTH][TILE_WIDTH];
__shared__ float nb[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x, tx = threadIdx.x;
int by = blockIdx.y, ty = threadIdx.y;

int row = by * TILE_WIDTH + ty;
int col = bx * TILE_WIDTH + tx;

float sum = 0;


//每个线程都会执行整个函数，因此每次都是不一样的(ty, tx)位置
for(int m = 0; m < n / TILE_WIDTH; m++){
na[ty][tx] = a[row * n + m * TILE_WIDTH + tx];
nb[ty][tx] = b[(ty + m * TILE_WIDTH) * n + col];
__syncthreads();
//整个tile的值都全了才能继续算

#pragma unroll TILE_WIDTH
for(int k = 0; k < TILE_WIDTH; k++){
sum += na[ty][k] * nb[k][tx];
}
__syncthreads();
//算完这一个tile才能再往里写
}
c[row * n + col] = sum;
}
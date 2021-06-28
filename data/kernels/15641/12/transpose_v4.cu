#include "includes.h"
__global__ void transpose_v4(float* a,float* b, int n){

int blockIdx_x = blockIdx.y;
int blockIdx_y = (blockIdx.x+blockIdx.y)%gridDim.x;

int tx = threadIdx.x;
int ty = threadIdx.y;

int bx = blockIdx_x;
int by = blockIdx_y;

int i = bx*BX + tx;
int j = by*BY + ty;

__shared__ float tile[BY][BX+1]; //Very slight modification to avoid bank conflict in shared mem

if(i >= n || j >= n) return;

tile[ty][tx] = a[j*n+i];

__syncthreads();

i = by*BY + tx;
j = bx*BX + ty;

b[j*n+i] = tile[tx][ty];

}
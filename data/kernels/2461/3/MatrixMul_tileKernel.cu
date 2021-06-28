#include "includes.h"
__global__ void MatrixMul_tileKernel(float* Md, float* Nd, float* Pd, int Width){

int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;
int tx = threadIdx.x, ty = threadIdx.y;
__shared__ float a[TILE_WIDTH][TILE_WIDTH], b[TILE_WIDTH][TILE_WIDTH];
float Pvalue = 0;
//Each thread computes one element of the block sub-matrix
for(int k=0; k< Width/TILE_WIDTH; k++){
a[ty][tx] = Md[Row*Width+k*TILE_WIDTH+tx];
b[ty][tx] = Nd[Col+Width*(k*TILE_WIDTH + ty)];
__syncthreads(); //sync all threads in a block;
for(int kk=0; kk<TILE_WIDTH; kk++)
Pvalue += a[ty][kk]*b[kk][tx];
__syncthreads(); //avoid memory hazards;
}
Pd[Row*Width+Col] = Pvalue;
}
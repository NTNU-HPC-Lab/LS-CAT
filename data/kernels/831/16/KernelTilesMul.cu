#include "includes.h"
__global__ void  KernelTilesMul(float *Mat1,float *Mat2,float *Mat3,int rowM1,int colM1,int colM2){

__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;

int row = by * TILE_WIDTH + ty;
int col = bx * TILE_WIDTH + tx;

float Pvalue = 0.0;


for(int k = 0; k < (colM1+TILE_WIDTH-1)/(TILE_WIDTH); ++k){

if(k*TILE_WIDTH + tx < colM1 && row < rowM1){
Mds[ty][tx] = Mat1[row*colM1 + k*TILE_WIDTH + tx];
}else{
Mds[ty][tx] = 0.0;
}
if(k*TILE_WIDTH + ty < colM1 && col < colM2){
Nds[ty][tx] = Mat2[(k*TILE_WIDTH + ty) * colM2 + col];
}else{
Nds[ty][tx] =0.0;
}

__syncthreads();

for(int k = 0; k < TILE_WIDTH; ++k){
Pvalue += Mds[ty][k] * Nds[k][tx];
}
__syncthreads();
}

if (row < rowM1 && col < colM2){
Mat3[row*colM2+col] = Pvalue;
}

}
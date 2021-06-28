#include "includes.h"


// CUDA runtime

// Utilities and system includes

__constant__ double dev_kernel[KERNEL_LENGTH*KERNEL_LENGTH];

__global__ void runConvolutionGPU(double* image, double* result, int height, int width, int step)
{
int tx = threadIdx.x;
int ty = threadIdx.y;
//int O_TILE_WIDTH = blockDim.x-(lkernel/2)*2;
//int O_TILE_HEIGHT = blockDim.y-(lkernel/2)*2;
int row_o = threadIdx.y + blockIdx.y*O_TILE_HEIGHT;
int col_o = threadIdx.x + blockIdx.x*O_TILE_WIDTH;

int row_i = row_o - KERNEL_LENGTH/2;
int col_i = col_o - KERNEL_LENGTH/2;

__shared__ double N_ds[BLOCK_DIM_Y][BLOCK_DIM_X];

if((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < height)){
N_ds[ty][tx] = image[row_i*width+col_i];
}else{
N_ds[ty][tx] = 0.0f;
}

__syncthreads();

double output = 0.0f;
if(tx%step ==0 && ty%step==0 && ty < O_TILE_HEIGHT && tx < O_TILE_WIDTH){
for(int i=0; i<KERNEL_LENGTH; i++){
for(int j=0; j<KERNEL_LENGTH; j++){
output += dev_kernel[i*KERNEL_LENGTH+j]*N_ds[(i+ty)][(j+tx)];
}
}
if(row_o < height && col_o < width){
result[(row_o/step)*width/step+col_o/step] = output/9.0;
}
}
}
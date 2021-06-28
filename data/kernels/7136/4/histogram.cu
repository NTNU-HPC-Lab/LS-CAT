#include "includes.h"
__global__ void histogram( int * hist_out, unsigned char * img_in, int img_w,int img_h,  int nbr_bin){

int tx=threadIdx.x;
int ty=threadIdx.y;
int bx=blockIdx.x;
int by=blockIdx.y;

unsigned int col= tx + blockDim.x * bx;

unsigned int row= ty + blockDim.y * by;

int grid_width = gridDim.x * blockDim.x;
int id = row * grid_width + col;

if(id<nbr_bin)
hist_out[id] = 0;

__syncthreads();

if(row<img_w && col<img_h)
atomicAdd( &(hist_out[img_in[id]]), 1);

}
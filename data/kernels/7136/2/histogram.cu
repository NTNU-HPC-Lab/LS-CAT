#include "includes.h"
__global__ void histogram( int * hist_out, unsigned char * img_in, int img_w,int img_h,  int nbr_bin){

int tx=threadIdx.x;
int ty=threadIdx.y;
int bx=blockIdx.x;
int by=blockIdx.y;

__shared__ int smem[256];
smem[threadIdx.x]=0;
__syncthreads();

unsigned int col= tx + blockDim.x * bx;
unsigned int row= ty + blockDim.y * by;


int grid_width = gridDim.x * blockDim.x;
int id = row * grid_width + col;

if(row<img_w && col<img_h)
atomicAdd( &(smem[img_in[id]]) ,1);

__syncthreads();

atomicAdd(&(hist_out[threadIdx.x]),smem[threadIdx.x]);


}
#include "includes.h"
__device__ int glb_hist[COLORS];   __global__ void calc_histogram(unsigned char * img_in, int offset_start, int offset_end){
int ix = blockIdx.x * blockDim.x + threadIdx.x;;
const int gridW = gridDim.x * blockDim.x;
int Row, pos;

__shared__ int hist[COLORS];

if (threadIdx.x < COLORS) {
hist[threadIdx.x] = 0;
}
__syncthreads();

int RowNum = (offset_end - offset_start) / gridW;
int extras = (offset_end - offset_start) % gridW;



for (Row=0; Row<RowNum; Row++) {
pos = Row * gridW + ix;
atomicAdd(&hist[img_in[pos + offset_start]],1);
}
if (extras && ix < extras) {
pos = Row * gridW + ix;
atomicAdd(&hist[img_in[pos + offset_start]],1);
}
__syncthreads();
if (threadIdx.x < COLORS) {
atomicAdd(&glb_hist[threadIdx.x],hist[threadIdx.x]);
}
}
__global__ void calc_histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){

int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;
const int gridW = gridDim.x * blockDim.x;
int img_position = iy * gridW + ix; //thesh mesa sthn eikona synarthsh tou gridW

// __shared__ hist_out[nbr_bin];

if (img_position < nbr_bin) { //allagh tou img_position me threadIdx.x se auto to if anevazei poly to contrast
hist_out[img_position] = 0;
}

__syncthreads();

if(img_position < img_size){
atomicAdd(&hist_out[img_in[img_position]],1);
}
__syncthreads();
}
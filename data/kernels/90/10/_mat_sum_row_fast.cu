#include "includes.h"
__global__ void _mat_sum_row_fast(float *m, float *target,int nrow, int ncol, int agg_col){
int tx = blockIdx.x * blockDim.x + threadIdx.x;

__shared__ float accum[NUM_THREAD_PER_ROW];

if(tx < ncol){
accum[threadIdx.x] = m[blockIdx.y*ncol+tx];
}else{
accum[threadIdx.x] = 0.0f;
}
__syncthreads();

if(NUM_THREAD_PER_ROW >= 512){
if(threadIdx.x < 256)
accum[threadIdx.x] += accum[threadIdx.x+256];
__syncthreads();
}

if(NUM_THREAD_PER_ROW >= 256){
if(threadIdx.x < 128)
accum[threadIdx.x] += accum[threadIdx.x+128];
__syncthreads();
}

//NUM_THREAD_PER_ROW at least 128
if(threadIdx.x < 64)
accum[threadIdx.x] += accum[threadIdx.x+64];
__syncthreads();

if(threadIdx.x < 32){
accum[threadIdx.x] += accum[threadIdx.x+32];
accum[threadIdx.x] += accum[threadIdx.x+16];
accum[threadIdx.x] += accum[threadIdx.x+8];
accum[threadIdx.x] += accum[threadIdx.x+4];
accum[threadIdx.x] += accum[threadIdx.x+2];
accum[threadIdx.x] += accum[threadIdx.x+1];
}
target[blockIdx.y*agg_col+blockIdx.x] = accum[0];
}
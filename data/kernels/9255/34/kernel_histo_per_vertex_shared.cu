#include "includes.h"
__global__ void kernel_histo_per_vertex_shared( unsigned int *ct, unsigned int *histo){
// get unique id for each thread in each block
unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
unsigned int tid_y = threadIdx.y + blockDim.y*blockIdx.y;

if( tid_x >= constant_n_test_vertices ) return;

unsigned int vertex_offset = tid_x*constant_n_hits;
unsigned int bin;
unsigned int stride = blockDim.y*gridDim.y;
unsigned int stride_block = blockDim.y;
unsigned int ihit = vertex_offset + tid_y;
unsigned int time_offset = tid_x*constant_n_time_bins;

unsigned int local_ihit = threadIdx.y;
extern __shared__ unsigned int temp[];
while( local_ihit<constant_n_time_bins ){
temp[local_ihit] = 0;
local_ihit += stride_block;
}

__syncthreads();

while( ihit<vertex_offset+constant_n_hits){

bin = ct[ihit];
atomicAdd(&temp[bin - time_offset],1);
ihit += stride;

}

__syncthreads();

local_ihit = threadIdx.y;
while( local_ihit<constant_n_time_bins ){
atomicAdd( &histo[local_ihit+time_offset], temp[local_ihit]);
local_ihit += stride_block;
}


}
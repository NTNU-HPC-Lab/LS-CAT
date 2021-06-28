#include "includes.h"
__global__ void kernel_histo_per_vertex( unsigned int *ct, unsigned int *histo){

// get unique id for each thread in each block
unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
unsigned int tid_y = threadIdx.y + blockDim.y*blockIdx.y;

if( tid_x >= constant_n_test_vertices ) return;

unsigned int vertex_offset = tid_x*constant_n_hits;
unsigned int bin;
unsigned int stride = blockDim.y*gridDim.y;
unsigned int ihit = vertex_offset + tid_y;

while( ihit<vertex_offset+constant_n_hits){

bin = ct[ihit];
//histo[bin]++;
atomicAdd( &histo[bin], 1);
ihit += stride;

}
__syncthreads();
}
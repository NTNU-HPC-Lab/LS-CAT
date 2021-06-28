#include "includes.h"
__global__ void kernel_histo_stride_2d( unsigned int *ct, unsigned int *histo){

// get unique id for each thread in each block
unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
unsigned int tid_y = threadIdx.y + blockDim.y*blockIdx.y;

unsigned int size = blockDim.x * gridDim.x;
unsigned int max = constant_n_hits*constant_n_test_vertices;

// map the two 2D indices to a single linear, 1D index
int tid = tid_y * size + tid_x;

/*
unsigned int vertex_index = (int)(tid/constant_n_time_bins);
unsigned int time_index = tid - vertex_index * constant_n_time_bins;

// skip if thread is assigned to nonexistent vertex
if( vertex_index >= constant_n_test_vertices ) return;

// skip if thread is assigned to nonexistent hit
if( time_index >= constant_n_time_bins ) return;

unsigned int vertex_block = constant_n_time_bins*vertex_index;

unsigned int vertex_block2 = constant_n_PMTs*vertex_index;
*/

unsigned int stride = blockDim.y * gridDim.y*size;

while( tid < max ){
atomicAdd( &histo[ct[tid]], 1);
tid += stride;
}


}
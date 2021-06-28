#include "includes.h"
__global__ void kernel_histo_one_thread_one_vertex( unsigned int *ct, unsigned int *histo ){


// get unique id for each thread in each block
unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;

unsigned int vertex_index = tid_x;
unsigned int bin ;
unsigned int max = constant_n_test_vertices*constant_n_hits;
unsigned int size = vertex_index * constant_n_hits;

for( unsigned int ihit=0; ihit<constant_n_hits; ihit++){
bin = size + ihit;
if( bin < max)
atomicAdd(&histo[ct[bin]],1);
}

}
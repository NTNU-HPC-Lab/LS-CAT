#include "includes.h"
__global__ void kernel_histo_stride( unsigned int *ct, unsigned int *histo){


int i = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while( i < constant_n_hits*constant_n_test_vertices ){
atomicAdd( &histo[ct[i]], 1);
i += stride;
}


}
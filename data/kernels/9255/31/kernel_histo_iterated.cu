#include "includes.h"
__global__ void kernel_histo_iterated( unsigned int *ct, unsigned int *histo, unsigned int offset ){


extern __shared__ unsigned int temp[];
unsigned int index = threadIdx.x + offset;
temp[index] = 0;
__syncthreads();
int i = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int size = blockDim.x * gridDim.x;
unsigned int max = constant_n_hits*constant_n_test_vertices;
while( i < max ){
atomicAdd( &temp[ct[i]], 1);
i += size;
}
__syncthreads();
atomicAdd( &(histo[index]), temp[index] );


}
#include "includes.h"
__global__ void convolution1d_tiles_constant_kernel(int *In, int *Out){
unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; // Index 1d iterator.
__shared__ int Tile[TILE_SIZE + Mask_size - 1];
int n = Mask_size/2;
int halo_left_index  = (blockIdx.x - 1 ) * blockDim.x + threadIdx.x;
if (threadIdx.x  >= blockDim.x - n ){
Tile[threadIdx.x - (blockDim.x - n )] = (halo_left_index < 0) ? 0 : In[halo_left_index];
}

if(index<N_elements){Tile[n + threadIdx.x] = In[index];
}else{Tile[n + threadIdx.x] = 0;}
int halo_right_index = (blockIdx.x + 1 ) * blockDim.x + threadIdx.x;
if (threadIdx.x < n) {
Tile[n + blockDim.x + threadIdx.x]=  (halo_right_index >= N_elements) ? 0 : In[halo_right_index];
}

__syncthreads();
int Value = 0;
for (unsigned int j = 0; j  < Mask_size; j ++) {
Value += Tile[threadIdx.x + j] * Global_Mask[j];
}
Out[index] = Value;
}
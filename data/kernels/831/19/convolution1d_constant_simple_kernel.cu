#include "includes.h"
__global__ void convolution1d_constant_simple_kernel(int *In, int *Out){

int i = blockIdx.x*blockDim.x + threadIdx.x;
__shared__ float N_ds[TILE_SIZE];
N_ds[threadIdx.x] = In[i];
__syncthreads();
int This_tile_start_point = blockIdx.x * blockDim.x;
int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
int N_start_point = i - (Mask_size/2);
int Pvalue = 0;
for (int j = 0; j < Mask_size; j ++){
int N_index = N_start_point + j;
if (N_index >= 0 && N_index < N_elements){
if ((N_index >= This_tile_start_point) && (N_index < Next_tile_start_point)){
Pvalue += N_ds[threadIdx.x+j-(Mask_size/2)]*Global_Mask[j];
} else{ Pvalue += In[N_index] * Global_Mask[j]; }
}
}
Out[i] = Pvalue;
}
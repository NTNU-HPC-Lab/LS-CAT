#include "includes.h"
__global__ void convolution1d_notile_noconstant_kernel(int *In, int *Out){
unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; // Index 1d iterator.
int Value = 0;
int N_start_point = index - (Mask_size/2);
for ( int j = 0; j  < Mask_size; j ++) {
if (N_start_point + j >= 0 && N_start_point + j < N_elements) {
Value += In[N_start_point + j] * Global_Mask[j];
}
}
Out[index] = Value;
}
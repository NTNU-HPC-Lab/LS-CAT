#include "includes.h"
__device__ void convolution(int conv_col, int conv_row, float *d_kernel, int k_size, float *d_matrix, int size_x, int size_y, float *d_conv, int max_row, int max_col){
int conv_index = conv_col+ conv_row*max_col;
d_conv[conv_index] = 0;
for(int k_row = 0;  k_row < k_size; k_row ++){
for(int k_col = 0;  k_col < k_size ; k_col ++){
d_conv[conv_index] +=
d_kernel[k_col + (k_row*k_size)] *
d_matrix[(conv_col+k_col) + (conv_row+k_row)*size_x];
//		printf("row %i col %i d_conv[] = %f \n", row, col, d_conv[col+ row*max_col]);
}
}
}
__global__ void valid_convolution(float *d_kernel, int k_size, float *d_matrix, int size_x, int size_y, float *d_conv, int max_row, int max_col){
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;

if(max_row > row && max_col > col){
convolution(col, row, d_kernel, k_size, d_matrix, size_x, size_y, d_conv, max_row, max_col);
}
}
#include "includes.h"
__global__ void valid_convolution(float *d_kernel, int k_size, float *d_matrix, int size_x, int size_y, float *d_conv, int max_row, int max_col){
int row = threadIdx.y + blockIdx.y * blockDim.y;
int col = threadIdx.x + blockIdx.x * blockDim.x;
if(max_row > row && max_col > col){
d_conv[col + row * max_col] = 0;
for(int k_row = 0;  k_row < k_size; k_row ++){
for(int k_col = 0;  k_col < k_size ; k_col ++){
d_conv[col + row * max_col] += d_kernel[k_col + (k_row*k_size)] * d_matrix[(col+k_col) + (row+k_row)*size_x];
//			printf("row %i col %i d_conv[] = %f \n", row, col, d_conv[col+ row*max_col]);
}
}
}
}
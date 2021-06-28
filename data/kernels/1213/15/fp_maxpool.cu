#include "includes.h"
__global__ void fp_maxpool(float* output, float* input, const int kernel_size, const int size, const int n_size, const int in_channel, bool SAME)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int totalPos = blockDim.x * gridDim.x;
const int N = kernel_size * kernel_size * n_size * n_size * in_channel;  // total number of connections in this convolution
const int padding = (kernel_size - 1) / 2;  // number of padding for both ends
int input_row, input_col;
// distribute certain number of connections to each thread regardless of detailed position and shape
for(int n = N * pos / totalPos; n < N * (pos+1) / totalPos; n++){
int idx = n;
const int i_kernel_row = ((idx /= 1	) % kernel_size);
const int i_kernel_col = ((idx /= kernel_size	) % kernel_size);
const int i_channel = ((idx /= kernel_size	) % in_channel);
const int i_row = ((idx /= in_channel	) % n_size);
const int i_col = ((idx /= n_size	) % n_size);
float maxidx = (float)-100;
// corresponding position of the input matrix and size of output matrix
if (SAME){ // SAME padding scheme implemented
input_row = i_kernel_row + i_row - padding;
input_col = i_kernel_col + i_col - padding;
}
else{
input_row = i_kernel_row + i_row;
input_col = i_kernel_col + i_col;
}
if(input_row >= 0 && input_row < size && input_col >=0 && input_col < size){
if (input[((i_channel % in_channel) * size + input_col) * size + input_row] > maxidx)
output[((i_channel % in_channel) * n_size + i_col) * n_size + i_row] = input[((i_channel % in_channel) * size + input_col) * size + input_row];
}
}
}
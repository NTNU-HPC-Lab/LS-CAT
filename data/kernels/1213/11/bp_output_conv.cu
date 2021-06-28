#include "includes.h"
__global__ void bp_output_conv(float *d_output, float *weight, float *nd_preact, const int size, const int kernel_size, const int n_size, const int in_channel, const int out_channel, bool CONV, bool SAME)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int totalPos = blockDim.x * gridDim.x;

const int N = kernel_size * kernel_size * size * size * in_channel * out_channel;
const int weight_channel = out_channel * in_channel;
const int padding = (kernel_size - 1) / 2;   // must be int

for (int n = N * pos / totalPos; n < N * (pos+1) / totalPos; ++n) {
int idx = n;
int bpinput_row;
int bpinput_col;
const int i_channel = ((idx /= 1	) % weight_channel);
const int i_kernel_row = ((idx /= weight_channel) % kernel_size);
const int i_kernel_col = ((idx /= kernel_size) % kernel_size);
const int i_row = ((idx /= kernel_size	) % size);
const int i_col = ((idx /= size) % size);

if (SAME){ // SAME padding scheme implemented
bpinput_row = i_kernel_row + i_row - padding;
bpinput_col = i_kernel_col + i_col - padding;
}
else{
bpinput_row = i_kernel_row + i_row - 2 * padding;
bpinput_col = i_kernel_col + i_col - 2 * padding;
}

if(bpinput_row >= 0 && bpinput_row < n_size && bpinput_col >=0 && bpinput_col < n_size){
atomicAdd(&d_output[((i_channel % in_channel) * size + i_col) * size + i_row],
weight[(i_channel * kernel_size + (kernel_size - 1 - i_kernel_col)) * kernel_size + kernel_size - 1 - i_kernel_row]
* nd_preact[((i_channel % out_channel) * n_size + bpinput_col) * n_size + bpinput_row]);
}
}
}
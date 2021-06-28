#include "includes.h"
__global__ void bcnn_cuda_im2col_kernel(const int n, const float* data_im, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, float *data_col)
{
int i, j, w, h, w_out, h_index, h_out, channel_in, channel_out;
int h_in, w_in;
int index = blockIdx.x * blockDim.x + threadIdx.x;
float *data_col_ptr = NULL;
const float *data_im_ptr = NULL;

for(; index < n; index += blockDim.x * gridDim.x) {
w_out = index % width_col;
h_index = index / width_col;
h_out = h_index % height_col;
channel_in = h_index / height_col;
channel_out = channel_in * ksize * ksize;
h_in = h_out * stride - pad;
w_in = w_out * stride - pad;
data_col_ptr = data_col;
data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
data_im_ptr = data_im;
data_im_ptr += (channel_in * height + h_in) * width + w_in;
for (i = 0; i < ksize; ++i) {
for (j = 0; j < ksize; ++j) {
h = h_in + i;
w = w_in + j;
*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
data_im_ptr[i * width + j] : 0;
data_col_ptr += height_col * width_col;
}
}
}
}
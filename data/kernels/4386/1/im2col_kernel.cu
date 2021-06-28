#include "includes.h"

__global__ void im2col_kernel(int n, float* data_im, int height, int width, int ksize_h, int ksize_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, int height_col, int width_col, float* data_col) {
for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < (n); index += blockDim.x * gridDim.x) {
int w_out = index % width_col;
index /= width_col;
int h_out = index % height_col;
int channel_in = index / height_col;
int channel_out = channel_in * ksize_h * ksize_w;
int h_in = h_out * stride_h - pad_h;
int w_in = w_out * stride_w - pad_w;
data_col += (channel_out * height_col + h_out) * width_col + w_out;
data_im += (channel_in * height + h_in) * width + w_in;
for (int i = 0; i < ksize_h; ++i) {
for (int j = 0; j < ksize_w; ++j) {
int h = h_in + i * dilation_h;
int w = w_in + j * dilation_w;
*data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
data_im[i * dilation_h * width + j * dilation_w] : 0;
data_col += height_col * width_col;
}
}
}
}
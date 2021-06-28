#include "includes.h"
__global__ void im2col_pad_kernel(float *im, int channels,  int height,  int width, int ksize,  int stride, float *data_col)
{
int c,h,w;
int height_col = 1 + (height-1) / stride;
int width_col = 1 + (width-1) / stride;
int channels_col = channels * ksize * ksize;

int pad = ksize/2;

int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
int col_size = height_col*width_col*channels_col;
if (id >= col_size) return;

int col_index = id;
w = id % width_col;
id /= width_col;
h = id % height_col;
id /= height_col;
c = id % channels_col;
id /= channels_col;

int w_offset = c % ksize;
int h_offset = (c / ksize) % ksize;
int im_channel = c / ksize / ksize;
int im_row = h_offset + h * stride - pad;
int im_col = w_offset + w * stride - pad;

int im_index = im_col + width*(im_row + height*im_channel);
float val = (im_row < 0 || im_col < 0 || im_row >= height || im_col >= width) ? 0 : im[im_index];

data_col[col_index] = val;
}
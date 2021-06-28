#include "includes.h"
__global__ void im2col_align_gpu_kernel(const int n, const float* data_im, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, float *data_col, const int bit_align)
{
//__shared__ float tmp_s[1];


int index = blockIdx.x*blockDim.x + threadIdx.x;
for (; index < n; index += blockDim.x*gridDim.x) {
int w_out = index % width_col;
int h_index = index / width_col;
int h_out = h_index % height_col;
int channel_in = h_index / height_col;
int channel_out = channel_in * ksize * ksize;
int h_in = h_out * stride - pad;
int w_in = w_out * stride - pad;
//float* data_col_ptr = data_col;
//float* data_col_ptr_32 = data_col + (channel_out * bit_align + h_out * width_col + w_out) / 32;
//data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
//data_col_ptr += channel_out * bit_align + h_out * width_col + w_out;
float* data_col_ptr = &data_col[channel_out * bit_align + h_out * width_col + w_out];
const float* data_im_ptr = data_im;
data_im_ptr += (channel_in * height + h_in) * width + w_in;
for (int i = 0; i < ksize; ++i) {
for (int j = 0; j < ksize; ++j) {
int h = h_in + i;
int w = w_in + j;

float val = (h >= 0 && w >= 0 && h < height && w < width) ?
data_im_ptr[i * width + j] : 0;

int pre_out_index = index % (width_col*height_col);
int out_index = (channel_out + i*ksize + j) * bit_align + pre_out_index;// h_out * width_col + w_out;
data_col[out_index] = val;

//(*data_col_ptr) = val;
//dst_s[threadIdx.x] = val;
//tmp_s[0] = val;

//(*data_col_ptr) = (h >= 0 && w >= 0 && h < height && w < width) ?
//    data_im_ptr[i * width + j] : 0;

//float src_val = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[i * width + j] : 0;
//unsigned int bit_mask = __ballot_sync(0xffffffff, src_val > 0);
//if (threadIdx.x % WARP_SIZE == 0) *((unsigned int*)data_col_ptr_32) = bit_mask;
// use atomicOr() // *dst_ptr |= (mask << (col_index % 8));
//data_col_ptr_32 += bit_align / 32;

//data_col_ptr += height_col * width_col;
data_col_ptr += bit_align;
}
}
}
}
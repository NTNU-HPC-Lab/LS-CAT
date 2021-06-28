#include "includes.h"
__global__ void _bcnn_backward_depthwise_sep_conv_data_kernel(int nthreads, float *dst_grad, float *weight_data, int batch_size, const int channels, int dst_h, int dst_w, const int src_h, const int src_w, int kernel_sz, int stride, int pad, float *src_grad)
{

int i, n, c, h, w, kw, kh, h_out_s, w_out_s, h_out, w_out, offset;
float value = 0.0f;
float *weight = NULL;

for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += blockDim.x * gridDim.x) {
n = i / channels / src_h / src_w;
c = (i / src_h / src_w) % channels;
h = (i / src_w) % src_h;
w = i % src_w;
weight = weight_data + c * kernel_sz * kernel_sz;
value = 0.0f;
for (kh = 0; kh < kernel_sz; ++kh) {
for (kw = 0; kw < kernel_sz; ++kw) {
h_out_s = h + pad - kh;
w_out_s = w + pad - kw;
if (((h_out_s % stride) == 0) && ((w_out_s % stride) == 0)) {
h_out = h_out_s / stride;
w_out = w_out_s / stride;
if ((h_out >= 0) && (h_out < dst_h) && (w_out >= 0) && (w_out < dst_w)) {
offset = ((n * channels + c) * dst_h + h_out) * dst_w + w_out;
value += (*weight) * dst_grad[offset];
}
}
++weight;
}
}
src_grad[i] += value;
}
}
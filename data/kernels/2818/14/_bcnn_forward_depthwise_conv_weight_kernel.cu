#include "includes.h"
__global__ void _bcnn_forward_depthwise_conv_weight_kernel( int nthreads, float *src_data, float *weight_data, int channels, int dst_h, int dst_w, int src_h, int src_w, int kernel_sz, int stride, int pad, float *dst_data) {
int i, n, c, h, w, kh, kw, h_in, w_in, offset;
float value;
float *weight = NULL;

for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads;
i += blockDim.x * gridDim.x) {
n = i / channels / dst_h / dst_w;
c = (i / dst_h / dst_w) % channels;
h = (i / dst_w) % dst_h;
w = i % dst_w;
weight = weight_data + c * kernel_sz * kernel_sz;
value = 0;
for (kh = 0; kh < kernel_sz; ++kh) {
for (kw = 0; kw < kernel_sz; ++kw) {
h_in = -pad + h * stride + kh;
w_in = -pad + w * stride + kw;
if ((h_in >= 0) && (h_in < src_h) && (w_in >= 0) &&
(w_in < src_w)) {
offset = ((n * channels + c) * src_h + h_in) * src_w + w_in;
value += (*weight) * src_data[offset];
}
++weight;
}
}
dst_data[i] = value;
}
}
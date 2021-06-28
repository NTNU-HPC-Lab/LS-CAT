#include "includes.h"
__global__ void bcnn_cuda_axpy_strided_kernel(int n, int num_batches, float a, float *x, float *y, int dst_stride, int src_stride, int x_c, int x_h, int x_w, int y_c, int y_h, int y_w, int min_c, int min_h, int min_w) {
int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
if (id >= n) {
return;
}
int i = id % min_w;
id /= min_w;
int j = id % min_h;
id /= min_h;
int k = id % min_c;
id /= min_c;
int b = id % num_batches;

int dst_int = i * dst_stride + y_w * (j * dst_stride + y_h * (y_c * b + k));
int src_ind = i * src_stride + x_w * (j * src_stride + x_h * (x_c * b + k));
y[dst_int] += a * x[src_ind];
}
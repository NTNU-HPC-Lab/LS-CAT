#include "includes.h"
__global__ void reorg_kernel(int N, float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i >= N) return;
int in_index = i;
int in_w = i%w;
i = i / w;
int in_h = i%h;
i = i / h;
int in_c = i%c;
i = i / c;
int b = i%batch;

int out_c = c / (stride*stride);

int c2 = in_c % out_c;
int offset = in_c / out_c;
int w2 = in_w*stride + offset % stride;
int h2 = in_h*stride + offset / stride;

int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

out[in_index] = x[out_index];
}
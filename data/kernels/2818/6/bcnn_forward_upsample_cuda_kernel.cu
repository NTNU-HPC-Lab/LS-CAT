#include "includes.h"
__global__ void bcnn_forward_upsample_cuda_kernel(size_t dst_sz, float *src, int w, int h, int c, int n, int size, float *dst) {
size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
if (i >= dst_sz) {
return;
}
int dst_idx = i;
int dst_w = i % (w * size);
i = i / (w * size);
int dst_h = i % (h * size);
i = i / (h * size);
int dst_c = i % c;
i = i / c;
int b = i % n;
int src_w = dst_w / size;
int src_h = dst_h / size;
int src_c = dst_c;

int src_idx = b * w * h * c + src_c * w * h + src_h * w + src_w;

dst[dst_idx] += src[src_idx];
}
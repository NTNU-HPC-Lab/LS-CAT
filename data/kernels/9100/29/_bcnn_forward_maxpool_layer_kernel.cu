#include "includes.h"
__global__ void _bcnn_forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, float *input, float *output, int *indexes)
{
int h = (in_h-1)/stride + 1;
int w = (in_w-1)/stride + 1;
int c = in_c;

int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (id >= n) {
return;
}

int j = id % w;
id /= w;
int i = id % h;
id /= h;
int k = id % c;
id /= c;
int b = id;

int out_index = j + w*(i + h*(k + c*b));
float max = -INFINITY;
int max_i = -1;
int l, m;
for (l = 0; l < size; ++l) {
for (m = 0; m < size; ++m) {
int cur_h = i * stride + l;
int cur_w = j * stride + m;
int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
int valid = (cur_h >= 0 && cur_h < in_h &&
cur_w >= 0 && cur_w < in_w);
float val = (valid != 0) ? input[index] : -INFINITY;
max_i = (val > max) ? index : max_i;
max   = (val > max) ? val   : max;
}
}
output[out_index] = max;
indexes[out_index] = max_i;
}
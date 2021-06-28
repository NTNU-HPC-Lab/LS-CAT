#include "includes.h"
__global__ void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *delta, float *prev_delta, int *indexes)
{
int h = (in_h + pad - size)/stride + 1;
int w = (in_w + pad - size)/stride + 1;
int c = in_c;
int area = (size-1)/stride;

int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if(id >= n) return;

int index = id;
int j = id % in_w;
id /= in_w;
int i = id % in_h;
id /= in_h;
int k = id % in_c;
id /= in_c;
int b = id;

int w_offset = -pad/2;
int h_offset = -pad/2;

float d = 0;
int l, m;
for(l = -area; l < area+1; ++l){
for(m = -area; m < area+1; ++m){
int out_w = (j-w_offset)/stride + m;
int out_h = (i-h_offset)/stride + l;
int out_index = out_w + w*(out_h + h*(k + c*b));
int valid = (out_w >= 0 && out_w < w &&
out_h >= 0 && out_h < h);
d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
}
}
prev_delta[index] += d;
}
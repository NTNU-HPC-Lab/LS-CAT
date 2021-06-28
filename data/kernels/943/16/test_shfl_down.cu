#include "includes.h"
__global__ void test_shfl_down(float *d_out, float *d_in, const int delta) {
float value = d_in[threadIdx.x];
value = __shfl_down_sync(0xffffffff,value,delta,16);
d_out[threadIdx.x] = value;
}
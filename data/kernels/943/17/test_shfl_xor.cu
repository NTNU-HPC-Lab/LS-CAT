#include "includes.h"
__global__ void test_shfl_xor(float *d_out, float *d_in, const int mask) {
float value = d_in[threadIdx.x];
value = __shfl_xor_sync(0xffffffff,value,mask,BDIMX);
d_out[threadIdx.x] = value;
}
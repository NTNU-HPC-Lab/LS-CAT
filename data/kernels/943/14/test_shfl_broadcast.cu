#include "includes.h"
__global__ void test_shfl_broadcast(float *d_out, float *d_in, const int srcLane) {
float value = d_in[threadIdx.x];
value = __shfl_sync(0xffffffff,value,srcLane,BDIMX);
d_out[threadIdx.x] = value;
}
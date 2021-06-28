#include "includes.h"
__global__ void prefixsum_combine(float* in, int in_length, float* out, int out_length){

int idx = blockDim.x * blockIdx.x + threadIdx.x;

if(idx < out_length && blockIdx.x > 0){
out[idx] += in[blockIdx.x - 1];
}

}
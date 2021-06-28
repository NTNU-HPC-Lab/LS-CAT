#include "includes.h"
__global__ void kernelBackprop1(float *delta_nabla_w,int w_off,float *activations,float *delta_nabla_b,int b_off) {
delta_nabla_w[w_off+(blockIdx.x*blockDim.x)+threadIdx.x]=activations[threadIdx.x]*delta_nabla_b[b_off+blockIdx.x];
//delta_nabla_w[w_off+(threadIdx.x*gridDim.x)+blockIdx.x]=activations[threadIdx.x]*delta_nabla_b[b_off+blockIdx.x];
}
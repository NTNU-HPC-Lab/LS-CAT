#include "includes.h"
__global__ void kernelUpdateWeights(float *nabla_w,float *weights,int tws,float eta,float mini_batch_size) {

float rate=eta/mini_batch_size;

if ((blockIdx.x*blockDim.x+threadIdx.x)<tws) {
weights[blockIdx.x*blockDim.x+threadIdx.x]-=rate*nabla_w[blockIdx.x*blockDim.x+threadIdx.x];
}
}
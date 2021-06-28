#include "includes.h"
__global__ void kernelFeedForward3(float *zs,float *biases,int b_off,float *activations) {
zs[(blockIdx.x*blockDim.x)+threadIdx.x]+=biases[b_off+threadIdx.x];
activations[(blockIdx.x*blockDim.x)+threadIdx.x]=1.0/(1.0+expf(-zs[(blockIdx.x*blockDim.x)+threadIdx.x]));
}
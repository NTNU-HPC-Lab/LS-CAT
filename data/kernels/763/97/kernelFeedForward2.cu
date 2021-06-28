#include "includes.h"
__global__ void kernelFeedForward2(float *zs,float *biases,int b_off,float *activations) {
zs[threadIdx.x]+=biases[b_off+threadIdx.x];
activations[threadIdx.x]=1.0/(1.0+expf(-zs[threadIdx.x]));
}
#include "includes.h"
__global__ void kernelFeedForward1b(float *zs,int bound,float *weights,int w_off,float *activations) {

int i;

zs[(blockIdx.x*blockDim.x)+threadIdx.x]=0.0;
for (i=0; i<bound; i++) {
zs[(blockIdx.x*blockDim.x)+threadIdx.x]+=weights[w_off+(threadIdx.x*bound)+i]*activations[(blockIdx.x*bound)+i];
}
}
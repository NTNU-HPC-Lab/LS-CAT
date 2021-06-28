#include "includes.h"
__global__ void kernelFeedForward1(float *zs,int bound2,float *weights,int w_off,float *activations1) {

int i;

zs[threadIdx.x]=0.0;
for (i=0; i<bound2; i++) {
zs[threadIdx.x]+=weights[w_off+(threadIdx.x*bound2)+i]*activations1[i];
}
}
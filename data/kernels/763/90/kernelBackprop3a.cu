#include "includes.h"
__global__ void kernelBackprop3a(float *delta_nabla_b,int b_off,int bound,int b_off_old,float *weights,int w_off_old) {

int j;

delta_nabla_b[b_off+threadIdx.x]=0.0;
for (j=0; j<bound; j++) {
delta_nabla_b[b_off+threadIdx.x]+=delta_nabla_b[b_off_old+j]*weights[w_off_old+(j*blockDim.x)+threadIdx.x];
}
}
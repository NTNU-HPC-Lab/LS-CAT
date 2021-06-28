#include "includes.h"
__global__ void kernelUpdateBiases(float *nabla_b,float *biases,float eta,float mini_batch_size) {

float rate=eta/mini_batch_size;

biases[threadIdx.x]-=rate*nabla_b[threadIdx.x];
}
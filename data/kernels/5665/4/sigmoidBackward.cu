#include "includes.h"
__device__ float sigmoid_derivate(float x){
return __fmul_rn(x, __fsub_rn(1.0f, x));
}
__device__ float sigmoid(float x){
return __frcp_rn(__fadd_rn(1, exp(-x)));
}
__global__ void sigmoidBackward(float* R, float* V, int x, int y){
int index = blockDim.x * blockIdx.x + threadIdx.x;
if(index < x*y)
R[index] = sigmoid_derivate(V[index]);
}
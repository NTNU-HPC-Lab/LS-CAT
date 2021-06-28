#include "includes.h"
__global__ void reluForward(float* R, float* V, int x, int y){
int index = blockDim.x * blockIdx.x + threadIdx.x;
if(index < x*y)
R[index] = fmaxf(V[index], 0);
}
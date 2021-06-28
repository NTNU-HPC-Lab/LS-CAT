#include "includes.h"
__global__ void reluBackward(float* dZ, float* top_diff, float* V, int x, int y){
int index = blockDim.x * blockIdx.x + threadIdx.x;
if(index < x*y){
if(V[index] > 0) {
dZ[index] = top_diff[index];
}else{
dZ[index] = 0;
}
}
}
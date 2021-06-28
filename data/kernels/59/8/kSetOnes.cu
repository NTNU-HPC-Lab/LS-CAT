#include "includes.h"
__global__ void kSetOnes(float *dest, int count){
for (int i = threadIdx.x; i < count; i += blockDim.x) {
dest[i] = 1;
}
}
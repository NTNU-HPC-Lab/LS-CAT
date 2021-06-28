#include "includes.h"
__global__ void gShift(float* out, const float* in, int length, int offset) {
for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
if(index < length) {
if(index - offset < 0 || index - offset >= length)
out[index] = 0;
else
out[index] = in[index - offset];
}
}
}
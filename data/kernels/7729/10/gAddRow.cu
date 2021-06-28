#include "includes.h"
__global__ void gAddRow(float* out, const float* in, int length) {
for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
if(index < length) {
out[index] = in[index] + out[index];
}
}
}
#include "includes.h"
__global__ void gSetSparse(float* out, const size_t* indices, const float* values, int length) {
for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
if(index < length) {
out[indices[index]] = values[index];
}
}
}
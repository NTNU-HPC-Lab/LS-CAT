#include "includes.h"
__device__ inline float stableLogit(float x) {
if(x >= 0) {
float z = expf(-x);
return 1.0 / (1.0 + z);
} else {
float z = expf(x);
return z / (1.0 + z);
}
}
__global__ void gHighwayForward(float* out, const float* in1, const float* in2, const float* t, size_t length) {
for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
if(index < length) {
float sigma = stableLogit(t[index]);
out[index] = in1[index] * sigma + in2[index] * (1.f - sigma);
}
}
}
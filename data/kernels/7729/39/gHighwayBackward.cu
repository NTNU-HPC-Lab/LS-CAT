#include "includes.h"
__device__ inline float stableSigmoid(float x) {
if(x >= 0) {
float z = expf(-x);
return 1.0 / (1.0 + z);
} else {
float z = expf(x);
return z / (1.0 + z);
}
}
__global__ void gHighwayBackward(float* out1, float* out2, float* outt, const float* in1, const float* in2, const float* t, const float* adj, size_t length) {
for(int bid = 0; bid < length; bid += blockDim.x * gridDim.x) {
int index = bid + blockDim.x * blockIdx.x + threadIdx.x;
if(index < length) {
float sigma = stableSigmoid(t[index]);
out1[index] = sigma * adj[index];
out2[index] = (1.f - sigma) * adj[index];
outt[index]
= sigma * (1.f - sigma) * (in1[index] - in2[index]) * adj[index];
}
}
}
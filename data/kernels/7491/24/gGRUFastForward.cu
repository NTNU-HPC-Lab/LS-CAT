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
__global__ void gGRUFastForward(float* out, const float* state, const float* xW, const float* sU, const float* b, const float* mask, size_t rows, size_t cols, bool final) {
for(int bid = 0; bid < rows; bid += gridDim.x) {
int j = bid + blockIdx.x;
if(j < rows) {
float m = !mask || mask[j];
float* rowOut = out + j * cols;
const float* rowState = state + j * cols;

const float* xWrow = xW + j * cols * 3;
const float* sUrow = sU + j * cols * 3;

for(int tid = 0; tid < cols; tid += blockDim.x) {
int i = tid + threadIdx.x;
if(i < cols) {
float r = stableSigmoid(xWrow[i] + sUrow[i] + b[i]);

int k = i + cols;

float z = stableSigmoid(xWrow[k] + sUrow[k] + b[k]);

int l = i + 2 * cols;
float h;
if(final)
h = tanhf(xWrow[l] + (sUrow[l] + b[l]) * r);
else
h = tanhf(xWrow[l] + sUrow[l] * r + b[l]);

float out = (1.0f - z) * h + z * rowState[i];
rowOut[i] = m * out + (1 - m) * rowState[i];
}
}
}
}
}
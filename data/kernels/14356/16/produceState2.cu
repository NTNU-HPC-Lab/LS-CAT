#include "includes.h"

__device__ float sigmoid(float x) {
return 1 / (1 + expf(-x));
}
__global__ void produceState2(const float* arguments, const int argsSize, const float* weights, const int* topology, const int topSize, float* outStates) {
const int tid = threadIdx.x;
const int dim = argsSize + topSize;
extern __shared__ float s[];
float* states = s;
bool* ready = (bool*)&states[dim];
__shared__ int counter[1];

int r = tid;
while(r < dim) {
ready[r] = false;
r += blockDim.x;
}


if (tid == 0) {
counter[tid] = argsSize;
}
if (tid < argsSize) {
states[tid] = arguments[tid];
ready[tid] = true;
}
__syncthreads();

while(counter[0] < dim) {
const int index = counter[0] + tid;
const int topIndex = index - argsSize;
if (topIndex < topSize) {
const int leftBorder = topology[topIndex*3];
const int rightBorder = topology[topIndex*3 + 1];
const int weightsStart = topology[topIndex*3 + 2];

if (rightBorder <= counter[0]) {
float sum = 0;
for (int i = leftBorder; i < rightBorder; i++) {
sum += states[i] * weights[weightsStart + i - leftBorder];
}

states[index] = sigmoid(sum);
ready[index] = true;
}
}
__syncthreads();

if (tid == 0) {
int total = counter[0];
for (int i = total; i < total + blockDim.x && i < dim; i++) {
if (ready[i]) {
counter[0]++;
}
}
}
__syncthreads();
}

int n = tid;
while(n < dim) {
outStates[n] = states[n];
n += blockDim.x;
}
}
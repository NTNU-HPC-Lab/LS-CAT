#include "includes.h"

__device__ float sigmoid(float x) {
return 1 / (1 + expf(-x));
}
__global__ void produceState3(const float* arguments, const int argsSize, const float* weights, const int* topology, const int topSize, float* outStates) {
const int tid = threadIdx.x;
const int dim = argsSize + topSize;
extern __shared__ float s[];
float* states = s;
int* iters = (int*)&states[dim];

if (tid < argsSize) {
states[tid] = arguments[tid];
iters[tid] = 1;
} else {
iters[tid] = 0;
}
__syncthreads();

while(iters[tid] * blockDim.x + tid < dim) {
const int index = iters[tid] * blockDim.x + tid;
const int topIndex = index - argsSize;
const int leftBorder = topology[topIndex*3];
const int rightBorder = topology[topIndex*3 + 1];
const int weightsStart = topology[topIndex*3 + 2];

bool canStart = true;
for (int i = leftBorder; i < rightBorder; i++) {
int threadId = i % blockDim.x;
int mustCounted = i / blockDim.x + 1;
if (iters[threadId] < mustCounted) {
canStart = false;
break;
}
}

if (canStart) {
float sum = 0;
for (int i = leftBorder; i < rightBorder; i++) {
sum += states[i] * weights[weightsStart + i - leftBorder];
}
states[index] = sigmoid(sum);
iters[tid]++;
}
__syncthreads();
}

__syncthreads();

int n = tid;
while(n < dim) {
outStates[n] = states[n];
n += blockDim.x;
}
}
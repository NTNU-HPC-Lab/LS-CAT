#include "includes.h"
__device__ int binarySearchForMultinomial(float* dist, int size, float val) {
int start = 0;
int end = size;

while (end - start > 0) {
int mid = start + (end - start) / 2;

float midVal = dist[mid];
if (midVal < val) {
start = mid + 1;
} else {
end = mid;
}
}

if (start == size) {
// No probability mass or precision problems; just return the
// first element
start = 0;
}

return start;
}
__global__ void sampleMultinomialWithReplacement(curandStateMtgp32* state, int totalSamples, float* dest, long distributions, int categories, float* normDistPrefixSum) {
// At the moment, each warp computes one sample value in the binary
// search due to divergence. It seems possible to compute multiple
// values and limit divergence though later on. However, no matter
// what, all block threads must participate in the curand_uniform
// call to update the generator state.

// The block determines the distribution for which we generate a point
for (long curDist = blockIdx.x;
curDist < distributions;
curDist += gridDim.x) {
for (int sampleBase = 0;
sampleBase < totalSamples; sampleBase += blockDim.y) {
// The warp determines the sample
int sample = sampleBase + threadIdx.y;

// All threads participate in this
float r = curand_uniform(&state[blockIdx.x]);

if (threadIdx.x == 0 && sample < totalSamples) {
// Find the bucket that a uniform sample lies in
int choice = binarySearchForMultinomial(
normDistPrefixSum + curDist * categories,
categories,
r);

// Torch indices are 1-based
dest[curDist * totalSamples + sample] = (float) choice + 1.0f;
}
}
}
}
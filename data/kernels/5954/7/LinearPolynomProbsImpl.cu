#include "includes.h"
__global__ void LinearPolynomProbsImpl( const float* features, int batchSize, const int* splits, const float* conditions, const int* polynomOffsets, int polynomCount, float lambda, float* probs, const int* origFIds) {
if (threadIdx.x < batchSize) {
int polynomId = blockIdx.x;

features += threadIdx.x;
probs += threadIdx.x;

while (polynomId < polynomCount) {
int offset = polynomOffsets[polynomId];
int nextOffset = polynomOffsets[polynomId + 1];
const int depth = nextOffset - offset;
const int origFId = origFIds[polynomId];

bool zeroProb = false;
for (int i = 0; i < depth; ++i) {
if (zeroProb) {
continue;
}

const float c = __ldg(conditions + offset + i);

const int f = __ldg(splits + offset + i);
const float x = __ldg(features + f * batchSize);

if (x <= c) {
zeroProb = true;
}
}

float prob = 0.0f;
if (!zeroProb) {
// TODO we store fID = -1 as our bias column, but it's a hack and we need to get rid of this
if (origFId != -1) {
prob = __ldg(features + origFId * batchSize);
} else {
prob = 1.0f;
}
}

probs[polynomId * batchSize] = prob;
polynomId += gridDim.x;
}
}
}
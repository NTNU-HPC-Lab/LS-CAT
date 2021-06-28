#include "includes.h"
__global__ void ExpProbPolynomProbsImpl( const float* features, int batchSize, const int* splits, const float* conditions, const int* polynomOffsets, int polynomCount, float lambda, float* probs) {
if (threadIdx.x < batchSize) {
int polynomId = blockIdx.x;

features +=  threadIdx.x;
probs += threadIdx.x;

while (polynomId < polynomCount) {
int offset = polynomOffsets[polynomId];
int nextOffset = polynomOffsets[polynomId + 1];
const int depth = nextOffset - offset;

float logProb = 0;
bool zeroProb = false;
for (int i = 0; i < depth; ++i) {
if (zeroProb) {
continue;
}

const int f = __ldg(splits + offset + i);
const float c = __ldg(conditions + offset + i);
const float x = __ldg(features + f * batchSize);

const float val = -lambda * x;
const float expVal = 1.0f - expf(val);

if (isfinite(log(expVal))) {
logProb += log(expVal);
} else {
zeroProb = true;
}
}

float prob = 0.0f;
if (!zeroProb) {
prob = expf(logProb);
}

probs[polynomId * batchSize] = prob;
polynomId += gridDim.x;
}
}
}
#include "includes.h"
__global__ void SigmoidProbPolynomProbsImpl( const float* features, int batchSize, const int* splits, const float* conditions, const int* polynomOffsets, int polynomCount, float lambda, float* probs) {
if (threadIdx.x < batchSize) {
int polynomId = blockIdx.x;

features +=  threadIdx.x;
probs += threadIdx.x;

while (polynomId < polynomCount) {
int offset = polynomOffsets[polynomId];
int nextOffset = polynomOffsets[polynomId + 1];
const int depth = nextOffset - offset;

//            bool isTrue = true;
float logProb = 0;
for (int i = 0; i < depth; ++i) {
const int f = __ldg(splits + offset + i);
const float c = __ldg(conditions + offset + i);
const float x = __ldg(features + f * batchSize);
const float val = -lambda * (x - c);
//                isTrue = x <= c? false : isTrue;
const float expVal = 1.0f + expf(val);

//            p( split = 1) = 1.0 / (1.0 + exp(-(x - c)))
//            c = 0, x= inf, p = 1.0 / (1.0 + exp(-inf) = 0
//            log(p) = -log(1.0 + exp(-(x - c))
const float isTrueLogProb = isfinite(expVal) ? log(expVal) : val;
logProb -= isTrueLogProb;
}
const float prob = expf(logProb);
//            const float prob = isTrue ? 1 : 0;//exp(logProb);
probs[polynomId * batchSize] = prob;
polynomId += gridDim.x;
}
}
}
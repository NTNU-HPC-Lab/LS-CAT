#include "includes.h"
__global__ void kMultiSoftmaxCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs, float* top5Probs, const int numCases, const int numOut, const int setSize) {
const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

if (tx < numCases) {
const int label = int(labels[tx]);
const float maxp = maxProbs[tx];
const float labelp = probs[label * numCases + tx];

labelLogProbs[tx] = __logf(labelp);

int numBiggerProbs = 0, numEqualsProbs = 0;
for (int i = 0; i < numOut; ++i) {
numBiggerProbs += probs[i * numCases + tx] > labelp;
numEqualsProbs += probs[i * numCases + tx] == labelp;
}

const int slotsLeft = setSize - numBiggerProbs;

top5Probs[tx] = slotsLeft <= 0.0f ? 0.0f : (numEqualsProbs <= slotsLeft ? 1.0f : float(slotsLeft) / numEqualsProbs);
correctProbs[tx] = labelp != maxp ? 0.0f : 1.0f / float(numEqualsProbs);
}
}
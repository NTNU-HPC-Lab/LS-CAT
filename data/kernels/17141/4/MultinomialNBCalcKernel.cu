#include "includes.h"
__device__ inline unsigned int RM_Index(unsigned int row, unsigned int col, unsigned int width) {
return (row * width + col);
}
__global__ void MultinomialNBCalcKernel(const float *d_data, const int *d_labels, float *feature_probs, float *class_priors, unsigned int n_samples_, unsigned int n_classes_, unsigned int n_features_) {

// Each thread will take care of one term for all docs
unsigned int tidx = threadIdx.x;
unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
unsigned int i = 0, row = 0;

if (feat_col < n_features_) { /* End condition check */

/* For each document / sample */
for (i = 0; i < n_samples_; ++i) {
row = d_labels[i];

// No race condition since each thread deals with one feature only
feature_probs[RM_Index(row, feat_col, n_features_)] +=
d_data[RM_Index(i, feat_col, n_features_)];

// WARNING: thread divergence :(
if (feat_col == 0) {
class_priors[row] += 1;
}
}
}
return;
}
#include "includes.h"
__device__ inline unsigned int RM_Index(unsigned int row, unsigned int col, unsigned int width) {
return (row * width + col);
}
__global__ void BernoulliNBLearnKernel(float *feature_probs, float *class_count_, const float *d_row_sums, unsigned int n_samples_, unsigned int n_classes_, unsigned int n_features_) {

// Each thread will take one term
unsigned int tidx = threadIdx.x;
unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
unsigned int i = 0;

if (feat_col < n_features_) { // End condition check
// For each label
for (i = 0; i < n_classes_; ++i) {
feature_probs[RM_Index(i, feat_col, n_features_)] /=
class_count_[i]; // d_row_sums[i];

if (feat_col == 0) {
class_count_[i] = class_count_[i] / (float)n_samples_;
}
}
}
}
#include "includes.h"
__device__ inline unsigned int RM_Index(unsigned int row, unsigned int col, unsigned int width) {
return (row * width + col);
}
__global__ void ComplementNBCalcKernel(const float *d_data, const int *d_labels, float *per_class_feature_sum_, float *per_feature_sum_, unsigned int n_samples_, unsigned int n_features_) {

// Each thread will take care of one term for all docs
unsigned int tidx = threadIdx.x;
unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
unsigned int i = 0, row = 0;

if (feat_col < n_features_) { // End condition check
// For each document / sample
for (i = 0; i < n_samples_; ++i) {
row = d_labels[i];

// No race condition since each thread deals with one feature only
// This is embarrasingly parallel
per_class_feature_sum_[RM_Index(row, feat_col, n_features_)] +=
d_data[RM_Index(i, feat_col, n_features_)];

per_feature_sum_[feat_col] += d_data[RM_Index(i, feat_col, n_features_)];
}
}
return;
}
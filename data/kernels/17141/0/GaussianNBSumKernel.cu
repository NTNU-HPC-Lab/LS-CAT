#include "includes.h"
__device__ inline unsigned int RM_Index(unsigned int row, unsigned int col, unsigned int width) {
return (row * width + col);
}
__global__ void GaussianNBSumKernel(const float *d_data, const int *d_labels, float *feature_means_, int *class_count_, unsigned int n_samples_, unsigned int n_classes_, unsigned int n_features_) {

// Each thread will take care of one feature for all training samples
unsigned int tidx = threadIdx.x;
unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
unsigned int i = 0, row = 0;

if (feat_col < n_features_) { /* End condition check */

for (i = 0; i < n_samples_; ++i) { /* For each training sample */
row = d_labels[i];

// No race condition since each thread deals with one feature only
feature_means_[RM_Index(row, feat_col, n_features_)] +=
d_data[RM_Index(i, feat_col, n_features_)];

// WARNING: thread divergence :/
if (feat_col == 0) {
class_count_[row] += 1;
}
}
}
return;
}
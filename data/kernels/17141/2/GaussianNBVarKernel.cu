#include "includes.h"
__device__ inline unsigned int RM_Index(unsigned int row, unsigned int col, unsigned int width) {
return (row * width + col);
}
__global__ void GaussianNBVarKernel(const float *d_data, const int *d_labels, const float *feature_means_, float *feature_vars_, const int *class_count_, const unsigned int n_samples_, const unsigned int n_classes_, const unsigned int n_features_) {

// Each thread will take care of one feature for all training samples
unsigned int tidx = threadIdx.x;
unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
unsigned int i = 0, row = 0;

// Calculate variances
if (feat_col < n_features_) {        /* End condition check */
for (i = 0; i < n_samples_; ++i) { /* For each sample */
row = d_labels[i];
feature_vars_[RM_Index(row, feat_col, n_features_)] +=
pow(d_data[RM_Index(i, feat_col, n_features_)] -
feature_means_[RM_Index(row, feat_col, n_features_)],
2);
}

// Calculate coefficients
for (i = 0; i < n_classes_; ++i) { /* For each class */
feature_vars_[RM_Index(i, feat_col, n_features_)] /= class_count_[i];
}
}
}
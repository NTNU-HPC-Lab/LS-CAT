#include "includes.h"
__device__ inline unsigned int RM_Index(unsigned int row, unsigned int col, unsigned int width) {
return (row * width + col);
}
__global__ void ComplementNBLearnKernel(float *feature_weights_, float *per_class_feature_sum_, float *per_feature_sum_, float *per_class_sum_, float all_sum_, unsigned int n_classes_, unsigned int n_features_) {
// Each thread will take one feature
unsigned int tidx = threadIdx.x;
int feat_col = tidx + (blockIdx.x * blockDim.x);

unsigned int i = 0;
float den_sum = 0;
float num_sum = 0;

if (feat_col < n_features_) {        /* Boundary check */
for (i = 0; i < n_classes_; ++i) { /* For each class */
den_sum = all_sum_ - per_class_sum_[i];
num_sum = per_feature_sum_[feat_col] -
per_class_feature_sum_[RM_Index(i, feat_col, n_features_)];

feature_weights_[RM_Index(i, feat_col, n_features_)] =
log(num_sum + 1.0) - log(den_sum + n_features_);
}
}
}
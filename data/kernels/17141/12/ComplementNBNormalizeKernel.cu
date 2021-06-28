#include "includes.h"
__device__ inline unsigned int RM_Index(unsigned int row, unsigned int col, unsigned int width) {
return (row * width + col);
}
__global__ void ComplementNBNormalizeKernel(float *feature_weights_, float *per_class_sum_, unsigned int n_classes_, unsigned int n_features_) {
// Each thread will take one feature
int feat_col = threadIdx.x + (blockIdx.x * blockDim.x);
unsigned int i = 0;

if (feat_col < n_features_) {        /* Boundary condition check */
for (i = 0; i < n_classes_; ++i) { // For each class
feature_weights_[RM_Index(i, feat_col, n_features_)] /= per_class_sum_[i];
}
}
}
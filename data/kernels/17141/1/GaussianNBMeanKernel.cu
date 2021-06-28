#include "includes.h"
__device__ inline unsigned int RM_Index(unsigned int row, unsigned int col, unsigned int width) {
return (row * width + col);
}
__global__ void GaussianNBMeanKernel(float *feature_means_, int *class_count_, float *class_priors_, unsigned int n_samples_, unsigned int n_classes_, unsigned int n_features_) {

// Each thread will take care of one feature for all training samples
unsigned int tidx = threadIdx.x;
unsigned int feat_col = tidx + (blockIdx.x * blockDim.x);
unsigned int i = 0;

if (feat_col < n_features_) { /* End condition check */

/* Calculate Means */
for (i = 0; i < n_classes_; ++i) { /* For each class */
feature_means_[RM_Index(i, feat_col, n_features_)] /= class_count_[i];

// WARNING: thread divergence
// Calculating Class priors
if (feat_col == 0) {
class_priors_[i] = (float)class_count_[i] / n_samples_;
}
}
}
}
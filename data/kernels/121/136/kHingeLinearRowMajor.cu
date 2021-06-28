#include "includes.h"
__global__ void kHingeLinearRowMajor(float* mat, float* labels, float* target, unsigned int width, unsigned int height, float margin) {
int image_id = blockIdx.x * blockDim.x + threadIdx.x;
if (image_id < height) {
mat += image_id;
target += image_id;
const int correct_label = (int)labels[image_id];
const float correct_label_score = mat[correct_label * height];
float sum = 0;
for (unsigned int i = 0; i < width; i++) {
float diff = margin + mat[i*height] - correct_label_score;
float grad = (diff > 0) ? 1 : 0;
target[i*height] = (i == correct_label) ? 0 : grad;
sum             += (i == correct_label) ? 0 : grad;
}
target[correct_label * height] = -sum;
}
}
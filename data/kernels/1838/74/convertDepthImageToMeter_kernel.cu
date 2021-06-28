#include "includes.h"
__global__ void convertDepthImageToMeter_kernel(float *d_depth_image_meter, const unsigned int *d_depth_image_millimeter, int n_rows, int n_cols) {

const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x < n_cols && y < n_rows) {
int ind = y * n_cols + x;
unsigned int depth = d_depth_image_millimeter[ind];
d_depth_image_meter[ind] =
(depth == 4294967295) ? nanf("") : (float)depth / 1000.0f;
}
}
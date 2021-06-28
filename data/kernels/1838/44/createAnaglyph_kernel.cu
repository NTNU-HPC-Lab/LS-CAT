#include "includes.h"
__global__ void createAnaglyph_kernel(uchar4 *out_image, const float *left_image, const float *right_image, int width, int height, int pre_shift) {
const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
const int x_right = x - pre_shift;
const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
uchar4 temp;

if (x < width && y < height) {

temp.x = left_image[__mul24(y, width) + x];

if (x_right > 0 && x_right < width) {
temp.y = right_image[__mul24(y, width) + x_right];
temp.z = temp.y;
} else {
temp.y = 0;
temp.z = 0;
}

temp.w = 255;

out_image[__mul24(y, width) + x] = temp;
}
}
#include "includes.h"
__device__ static float rgbaToGray(uchar4 rgba) {
return (0.299f * (float)rgba.x + 0.587f * (float)rgba.y +
0.114f * (float)rgba.z);
}
__global__ void createAnaglyph_kernel(uchar4 *out_image, const uchar4 *left_image, const uchar4 *right_image, int width, int height, int pre_shift) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int x_right = x - pre_shift;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
uchar4 temp;

if (x < width && y < height) {

temp.x = rgbaToGray(left_image[y * width + x]);

if (x_right > 0 && x_right < width) {
temp.y = rgbaToGray(right_image[y * width + x_right]);
temp.z = temp.y;
} else {
temp.y = 0;
temp.z = 0;
}

temp.w = 255;

out_image[y * width + x] = temp;
}
}
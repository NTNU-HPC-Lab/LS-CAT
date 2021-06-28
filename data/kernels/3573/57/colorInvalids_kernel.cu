#include "includes.h"
__global__ void colorInvalids_kernel(uchar4 *out_image, const float *in_image, int width, int height) {
const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

if (x < width && y < height) {
int ind = __mul24(y, width) + x;
uchar4 temp = out_image[ind];
float value = in_image[ind];

if (!isfinite(value)) { // color
temp.x *= 0.5f;
temp.y *= 0.5f;
}

out_image[ind] = temp;
}
}
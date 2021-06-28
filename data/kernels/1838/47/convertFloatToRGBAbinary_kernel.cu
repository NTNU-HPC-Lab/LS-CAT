#include "includes.h"
__global__ void convertFloatToRGBAbinary_kernel(uchar4 *out_image, const float *in_image, int width, int height, float lowerLim, float upperLim) {
const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
uchar4 temp;
if (x < width && y < height) {
float val = in_image[__mul24(y, width) + x];

// draw everything invalid or out of lim in white
if (!isfinite(val) || (val < lowerLim) || (val > upperLim)) {
temp.x = 255;
temp.y = 255;
temp.z = 255;
temp.w = 255;
} else {
temp.x = 0.0f;
temp.y = 0.0f;
temp.z = 0.0f;
temp.w = 0.0f;
}
out_image[__mul24(y, width) + x] = temp;
}
}
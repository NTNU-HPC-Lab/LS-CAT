#include "includes.h"
__global__ void convertPitchedFloatToRGBA_kernel(uchar4 *out_image, const float *in_image, int width, int height, int pitch, float lowerLim, float upperLim) {
const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
uchar4 temp;
if (x < width && y < height) {
float val = *((float *)((char *)in_image + y * pitch) + x);

// first draw unmatched pixels in white
if (!isfinite(val)) {
temp.x = 255;
temp.y = 255;
temp.z = 255;
temp.w = 255;
} else {
// rescale value from [lowerLim,upperLim] to [0,1]
val -= lowerLim;
val /= (upperLim - lowerLim);

float r = 1.0f;
float g = 1.0f;
float b = 1.0f;
if (val < 0.25f) {
r = 0;
g = 4.0f * val;
} else if (val < 0.5f) {
r = 0;
b = 1.0 + 4.0f * (0.25f - val);
} else if (val < 0.75f) {
r = 4.0f * (val - 0.5f);
b = 0;
} else {
g = 1.0f + 4.0f * (0.75f - val);
b = 0;
}
temp.x = 255.0 * r;
temp.y = 255.0 * g;
temp.z = 255.0 * b;
temp.w = 255;
}
out_image[__mul24(y, width) + x] = temp;
}
}
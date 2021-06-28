#include "includes.h"
__global__ void convertPitchedFloatToGrayRGBA_kernel(uchar4 *out_image, const float *in_image, int width, int height, int pitch, float lowerLim, float upperLim) {
const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

uchar4 temp;

if (x < width && y < height) {
//    float val = in_image[__mul24(y,pitch)+x];
float val = *((float *)((char *)in_image + y * pitch) + x);

// rescale value from [lowerLim,upperLim] to [0,255]
val -= lowerLim;
val /= (upperLim - lowerLim);
val *= 255.0;

temp.x = val;
temp.y = val;
temp.z = val;
temp.w = 255;

out_image[__mul24(y, width) + x] = temp;
}
}
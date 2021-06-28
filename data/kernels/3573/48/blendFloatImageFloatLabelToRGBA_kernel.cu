#include "includes.h"
__global__ void blendFloatImageFloatLabelToRGBA_kernel( uchar4 *out_image, const float *in_image, const float *label, int width, int height, float lowerLim, float upperLim) {
const int x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
const int y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
uchar4 temp;
if (x < width && y < height) {
unsigned char img =
(unsigned char)(0.5f * in_image[__mul24(y, width) + x] + 128.0f);
float val = label[__mul24(y, width) + x];

// draw everything invalid or out of lim in white
if (!isfinite(val) || (val < lowerLim) || (val > upperLim)) {
// don't blend

temp.x = img;
temp.y = img;
temp.z = img;
temp.w = 255;

} else {

// blend

temp.x = 0.6f * img;
temp.y = 0.6f * img;
temp.z = img;
temp.w = 255;
}
out_image[__mul24(y, width) + x] = temp;
}
}
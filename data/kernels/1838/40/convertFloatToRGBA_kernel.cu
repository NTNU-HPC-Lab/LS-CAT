#include "includes.h"
__global__ void convertFloatToRGBA_kernel(uchar4 *out_image, const float *in_image, int width, int height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
uchar4 temp;
if (x < width && y < height) {
int IND = y * width + x;
float val = in_image[IND];
temp.x = val;
temp.y = val;
temp.z = val;
temp.w = 255;
out_image[IND] = temp;
}
}
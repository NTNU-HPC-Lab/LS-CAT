#include "includes.h"
__global__ void onBothBufferOperatorKernel(const int warpWidth, const int input0OffsetX, const int input0OffsetY, const int input0Width, const int input0Height, const uint32_t* input0Buffer, const int input1OffsetX, const int input1OffsetY, const int input1Width, const int input1Height, const uint32_t* input1Buffer, const int outputOffsetX, const int outputOffsetY, const int outputWidth, const int outputHeight, uint32_t* outputMask) {
// calculate normalized texture coordinates
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x < outputWidth && y < outputHeight) {
uint32_t v = 0;
const int outputX = x + outputOffsetX;
const int outputY = y + outputOffsetY;
const int input0X = (outputX + warpWidth - input0OffsetX) % warpWidth;
const int input0Y = (outputY - input0OffsetY);
const int input1X = (outputX + warpWidth - input1OffsetX) % warpWidth;
const int input1Y = (outputY - input1OffsetY);
if (input1X >= 0 && input1X < input1Width && input1Y >= 0 && input1Y < input1Height && input0X >= 0 &&
input0X < input0Width && input0Y >= 0 && input0Y < input0Height) {
if (input0Buffer[input0Y * input0Width + input0X] > 0 && input1Buffer[input1Y * input1Width + input1X] > 0) {
v = 1;
} else {
v = 0;
}
}
outputMask[y * outputWidth + x] = v;
}
}
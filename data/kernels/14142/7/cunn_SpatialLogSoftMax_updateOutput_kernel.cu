#include "includes.h"
__global__ void cunn_SpatialLogSoftMax_updateOutput_kernel(float *output, float *input, int classSize, int height, int width)
{
int batchIndex = blockIdx.x;
int index = threadIdx.x;

while (index < height*width) {
int y = index / width;
int x = index % width;
if (y >= height)
break;

// calculate input starting index in cuda layout (B x H x W x C)
int inputStartIndex =
(height*width*classSize)*batchIndex +
(width*classSize)*y +
(classSize)*x;

float sum = 0;
for (int i = 0; i < classSize; i++) {
sum += __expf(input[inputStartIndex + i]);
}
sum = 1.0f / sum;

for (int i = 0; i < classSize; i++) {
// calculate output index in torch layout (B x C x H x W)
int outputIndex =
(classSize*height*width)*batchIndex +
(height*width)*i +
(width)*y +
x;
output[outputIndex] = logf(sum * __expf(input[inputStartIndex + i]));
}
index += blockDim.x;
}
}
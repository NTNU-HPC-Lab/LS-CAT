#include "includes.h"
__global__ void cunn_SpatialLogSoftMax_updateGradInput_kernel(float *gradInput, float *output, float *gradOutput, int classSize, int height, int width)
{
int batchIndex = blockIdx.x;
int index = threadIdx.x;

while (index < height*width) {
int y = index / width;
int x = index % width;
if (y >= height)
break;

// calculate output starting index in cuda layout (B x H x W x C)
int outputStartIndex =
(height*width*classSize)*batchIndex +
(width*classSize)*y +
(classSize)*x;

float sum = 0;
for (int i = 0; i < classSize; i++) {
sum += gradOutput[outputStartIndex + i];
}

for (int i = 0; i < classSize; i++) {
// calculate input index in torch layout (B x C x H x W)
int inputIndex =
(classSize*height*width)*batchIndex +
(height*width)*i +
(width)*y +
x;
gradInput[inputIndex] = gradOutput[outputStartIndex + i] - __expf(output[outputStartIndex + i]) * sum;
}
index += blockDim.x;
}
}
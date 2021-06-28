#include "includes.h"
__global__ void add_weighted_kernel(unsigned int batchSize, unsigned int nbOutputs, unsigned int outputsHeight, unsigned int outputsWidth, float* estimated_labels, unsigned int nbChannels, unsigned int image_height, unsigned int image_width, float* input_image, unsigned char* workspace, float alpha)
{
const int batchEstimatedOffset = nbOutputs * outputsHeight * outputsWidth * blockIdx.z;
const int batchImageOffset = nbChannels * image_height * image_width * blockIdx.z;

const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < outputsWidth * outputsHeight; i += stride)
{
unsigned int outputMax = 0;

if (nbOutputs > 1)
{
float maxVal = estimated_labels[i + batchEstimatedOffset];

for (unsigned int cls = 1; cls < nbOutputs; ++cls) {
const float tmp = estimated_labels[i
+ cls*outputsWidth*outputsHeight
+ batchEstimatedOffset];

if (tmp > maxVal) {
outputMax = cls;
maxVal = tmp;
}
}
const unsigned char ch0
= (unsigned char) max(colors[outputMax%4][0]*alpha, min(255.0, colors[outputMax%4][0]*alpha + input_image[i + batchImageOffset]));
const unsigned char ch1
= (unsigned char) max(colors[outputMax%4][1]*alpha, min(255.0, colors[outputMax%4][1]*alpha + input_image[i + image_height*image_width + batchImageOffset]));
const unsigned char ch2
= (unsigned char) max(colors[outputMax%4][2]*alpha, min(255.0, colors[outputMax%4][2]*alpha + input_image[i + 2*image_height*image_width + batchImageOffset]));

workspace[i*3 + batchImageOffset] = ch0;
workspace[i*3 + 1 + batchImageOffset] = ch1;
workspace[i*3 + 2 + batchImageOffset] = ch2;
}
}
}
#include "includes.h"
__global__ void cudaSBilinearTF_Forward_kernel( unsigned int outputWidth, unsigned int outputHeight, unsigned int nbChannels, unsigned int batchSize, unsigned int inputWidth, unsigned int inputHeight, const unsigned int* yLowIdx, const unsigned int* yHighIdx, const float* yInter, const unsigned int* xLowIdx, const unsigned int* xHighIdx, const float* xInter, const float* input, float* outputs)
{

const unsigned int inputOffset
= (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels*inputWidth*inputHeight;

const unsigned int outputOffset
= (blockIdx.z * blockDim.z + threadIdx.z) * nbChannels*outputWidth*outputHeight;
for (unsigned int ch = blockIdx.x; ch < nbChannels; ch += gridDim.x)
{
for (unsigned int oy = threadIdx.y; oy < outputHeight; oy += blockDim.y)
{
for (unsigned int ox = threadIdx.x; ox < outputWidth; ox += blockDim.x)
{
const unsigned int indexTL = xLowIdx[ox] + yLowIdx[oy]*inputWidth
+ ch*inputWidth*inputHeight
+ inputOffset;

const unsigned int indexTR = xHighIdx[ox] + yLowIdx[oy]*inputWidth
+ ch*inputWidth*inputHeight
+ inputOffset;

const unsigned int indexBL = xLowIdx[ox] + yHighIdx[oy]*inputWidth
+ ch*inputWidth*inputHeight
+ inputOffset;

const unsigned int indexBR = xHighIdx[ox] + yHighIdx[oy]*inputWidth
+ ch*inputWidth*inputHeight
+ inputOffset;

const float top_left = input[indexTL];
const float top_right = input[indexTR];
const float bottom_left = input[indexBL];
const float bottom_right = input[indexBR];

const float top = top_left + (top_right - top_left) * xInter[ox];
const float bottom = bottom_left + (bottom_right - bottom_left) * xInter[ox];

outputs[ ox + oy*outputWidth
+ ch*outputWidth*outputHeight + outputOffset]  = top + (bottom - top) * yInter[oy];

}
}
}
}
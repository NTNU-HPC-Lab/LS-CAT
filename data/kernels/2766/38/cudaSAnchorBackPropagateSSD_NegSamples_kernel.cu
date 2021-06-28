#include "includes.h"
__global__ void cudaSAnchorBackPropagateSSD_NegSamples_kernel(const float* inputCls, float* diffOutputsCls, const float* confSamples, const int* keySamples, const int nbSamples, const int nbPositive, const unsigned int nbAnchors, const unsigned int outputsHeight, const unsigned int outputsWidth, const unsigned int batchSize)
{

const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

if(index < nbSamples)
{
const int indexSamples = keySamples[index];
const float error = inputCls[indexSamples];
//diffOutputsCls[indexSamples] = -inputCls[index] / (nbPositive * batchSize);
//printf("error[%d]: %f\n", indexSamples, error);

diffOutputsCls[indexSamples] = -error / (nbPositive * batchSize);
}

}
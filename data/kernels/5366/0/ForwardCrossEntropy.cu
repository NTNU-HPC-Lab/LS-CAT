#include "includes.h"
__global__ void ForwardCrossEntropy(float *output, float *labels, int nColsOutput, float *loss)
{
int col = blockIdx.x;

float temp = -(labels[col] * logf(output[col]) + logf(1 - output[col])
* (1 - labels[col]));
atomicAdd(loss, temp);
}
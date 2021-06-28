#include "includes.h"
__global__ void BackwardCrossEntropy(float *output, float *labels, int nColsOutput, float *dOutput)
{
int col = blockIdx.x;

dOutput[col] = (labels[col] / output[col] - (1 - labels[col]) /
(1 - output[col])) * -1;
}
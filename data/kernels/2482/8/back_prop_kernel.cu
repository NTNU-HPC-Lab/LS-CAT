#include "includes.h"
__global__ void back_prop_kernel(float *device_output, float *inP, float *m_hidden, float* weights_2, float* o_errG, int nInput, int nHidden, int nOutput,  float l_R)
{
int linearThreadIndex = threadIdx.x;

int unit = blockIdx.x;

__shared__ float weightedSum[1];

if (linearThreadIndex==0)
{
for (int i=0; i<nOutput; i++)
{

weightedSum[0] += weights_2[unit*nOutput + i] * o_errG[i];

}

}

__syncthreads();

if (linearThreadIndex < nInput)
{

device_output[linearThreadIndex*nHidden + unit] = l_R * inP[linearThreadIndex] * m_hidden[unit]*(1 - m_hidden[unit]) * weightedSum[0];

}

}
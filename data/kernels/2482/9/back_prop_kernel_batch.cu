#include "includes.h"
__global__ void back_prop_kernel_batch(float *device_output, float *inP, float *m_hidden, float* weights_2, float* o_errG, int nInput, int nHidden, int nOutput, float l_R, int batchSize)
{
int linearThreadIndex = threadIdx.x;

int unit = blockIdx.x%nHidden;

int batch = blockIdx.x/nHidden;

__shared__ float weightedSum[1];

float temp = 0.0;

if (linearThreadIndex ==0 && unit<nHidden)
{
for (int i=0; i<nOutput; i++)
{

weightedSum[0] += weights_2[unit*nOutput + i] * o_errG[batch*(nOutput+1) +i];

}

}

__syncthreads();

if (linearThreadIndex < nInput)
{
temp = l_R * inP[batch*(nInput+1) + linearThreadIndex] * m_hidden[batch*(nHidden+1) + unit]*(1 - m_hidden[batch*(nHidden+1) + unit]) * weightedSum[0];

atomicAdd(&device_output[linearThreadIndex*nHidden + unit], temp);

}


}
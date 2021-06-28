#include "includes.h"
__global__ void multiplyNumbersGPU(float *pDataA, float *pDataB, float *pResult)
{
int tid = (blockIdx.y * 128 * 256) + blockIdx.x * 256 + threadIdx.x;
pResult[tid] = sqrt(pDataA[tid] * pDataB[tid] / 12.34567) * sin(pDataA[tid]);

}
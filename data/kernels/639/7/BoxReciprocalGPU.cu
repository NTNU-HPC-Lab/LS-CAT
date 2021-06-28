#include "includes.h"
__global__ void BoxReciprocalGPU(double *gpu_prefact, double *gpu_sumRnew, double *gpu_sumInew, double *gpu_energyRecip, int imageSize)
{
int threadID = blockIdx.x * blockDim.x + threadIdx.x;
if(threadID >= imageSize)
return;

gpu_energyRecip[threadID] = ((gpu_sumRnew[threadID] * gpu_sumRnew[threadID] +
gpu_sumInew[threadID] * gpu_sumInew[threadID]) *
gpu_prefact[threadID]);
}
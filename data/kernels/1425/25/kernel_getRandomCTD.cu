#include "includes.h"
__global__ void kernel_getRandomCTD(double* dev_nt, double* dev_tran, double* dev_nr, double* dev_ramR, unsigned int out, int rSize, int tSize )
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;
float myrand;

curandState s;
curand_init(out, tid, 0, &s);

//myrand = curand_uniform(&s);
//myrand *= (0 - nC);
//myrand += (nC - 0);
//dev_ramC[tid] = (int)truncf(myrand);

myrand = curand_uniform(&s);
myrand *= (0 - tSize);
myrand += (tSize - 0);
int t = ((int)truncf(myrand) + blockIdx.x * tSize) * 2;
//int t = (blockIdx.x * tSize) * 2;
for (int n = 0; n < 2; n++)
{
dev_tran[tid * 2 + n] = dev_nt[t + n];
}

myrand = curand_uniform(&s);
myrand *= (0 - rSize);
myrand += (rSize - 0);
int r = ((int)truncf(myrand) + blockIdx.x * rSize) * 4;
//int r = (blockIdx.x + blockIdx.x * rSize) * 4;
for (int n = 0; n < 4; n++)
{
dev_ramR[tid * 4 + n] = dev_nr[r + n];
}
}
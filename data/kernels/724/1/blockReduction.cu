#include "includes.h"
__global__ void blockReduction(double* dN_pTdpTdphidy_d, int final_spectrum_size, int blocks_ker1)
{
long idx = threadIdx.x + blockDim.x * blockIdx.x;
if (idx < final_spectrum_size)
{
if (blocks_ker1 == 1) return; //Probably will never happen, but best to be careful
//Need to start at i=1, since adding everything to i=0
for (int i = 1; i < blocks_ker1; i++)
{
dN_pTdpTdphidy_d[idx] += dN_pTdpTdphidy_d[idx + i * final_spectrum_size];
if (isnan(dN_pTdpTdphidy_d[idx])) printf("found dN_pTdpTdphidy_d nan \n");
}
}
}
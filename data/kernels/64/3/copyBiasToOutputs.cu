#include "includes.h"
__global__ void copyBiasToOutputs(float *ptrbias, float *ptroutput, const int size1, const int size2, const int nOutputPlane, const int linestride, const int imstride)
{
// each thread has a value to manage...
//const int blk =blockDim.x;
const int tidx=blockDim.x*blockIdx.x + threadIdx.x;
const int tidy=blockIdx.y;
const int tidz=blockIdx.z;

float val = ptrbias[tidx];
ptroutput+= tidz*imstride + tidy*linestride;

for(int k=0; k<size2; k++)
{
if(tidx<nOutputPlane) {
ptroutput[k*nOutputPlane+tidx]=val;
}
}
}
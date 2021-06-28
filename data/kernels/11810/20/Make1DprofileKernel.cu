#include "includes.h"
__global__ void Make1DprofileKernel (double *gridfield, double *axifield, int nsec, int nrad)
{
int i = threadIdx.x + blockDim.x*blockIdx.x;
int j;

if (i < nrad){
double sum = 0.0;

for (j = 0; j < nsec; j++)
sum += gridfield[i*nsec + j];

axifield[i] = sum/(double)nsec;
}
}
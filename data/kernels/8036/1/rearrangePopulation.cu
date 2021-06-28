#include "includes.h"
__global__ void rearrangePopulation(float *gene, float *fit, int* metaData)
{
const int idx = threadIdx.x + blockDim.x*blockIdx.x;
int nGene = metaData[1];
int nHalf = nGene / 2;
if(idx> nHalf) return;

int j = nGene - 1 - idx;

if (fit[idx] < fit[j]) {
for(int k=0; k<6; k++) {
float t = gene[idx*6+k];
gene[idx*6+k] = gene[j*6+k];
gene[j*6+k] = t;
t = fit[idx];
fit[idx] = fit[j];
fit[j] = t;
}
}
}
#include "includes.h"
__global__ void rearrangePopulationWithRange(float *gene, float *fit, int *range)
{
const int idx = threadIdx.x + blockDim.x*blockIdx.x;
if(range[0]>range[1]) return;

int totalElements = range[1] - range[0] + 1;
int nHalf = totalElements / 2;
if(idx> nHalf) return;

int i = range[0] + idx;
int j = range[1] - idx;

if (fit[i] < fit[j]) {
for(int k=0; k<6; k++) {
float t = gene[i*6+k];
gene[i*6+k] = gene[j*6+k];
gene[j*6+k] = t;

}
float t = fit[i];
fit[i] = fit[j];
fit[j] = t;
}
}
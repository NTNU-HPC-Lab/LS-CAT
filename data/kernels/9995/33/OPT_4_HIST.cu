#include "includes.h"
__global__ void OPT_4_HIST(int *d_lcmMatrix, int *d_LCMSize, int *d_histogram, int n_vertices)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
int count = 0, countMax = -1;

if(i<n_vertices)
{
int iStart = 0;
if(i>0)
iStart = d_LCMSize[i - 1]; //Offset
count = 0;
int iSize = d_LCMSize[i] - iStart;

for(int j = 0; j < n_vertices; j++) {
int jStart = 0;
if(j>0)
jStart = d_LCMSize[j - 1]; //Offset

int jSize = d_LCMSize[j] - jStart;
if(iSize != jSize)
continue;

int eq = 1;
for(int k = 0; k < iSize; k++)
{
if(d_lcmMatrix[iStart + k] != d_lcmMatrix[jStart + k])
{
eq = 0;
break;
}
}
if(eq == 1)
{
count++;
}
}

if(countMax < count)
countMax = count;
atomicAdd((int*)&d_histogram[count], 1);
// d_histogram[count]++;
}
}
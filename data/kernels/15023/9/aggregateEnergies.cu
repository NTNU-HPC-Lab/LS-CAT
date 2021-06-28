#include "includes.h"
__global__ void aggregateEnergies(double *energies, int numEnergies, int interval, int batchSize)
{
int idx = batchSize * interval * (blockIdx.x * blockDim.x + threadIdx.x), i;

for (i = 1; i < batchSize; i++)
{
if (idx + i * interval < numEnergies)
{
energies[idx] += energies[idx + i * interval];
energies[idx + i * interval] = 0;
}
}
}
#include "includes.h"
__global__ void findCentroidsAtomicFreeReduce_64(int afLocal, int* responses, int nPixels, int* cluster, int* centroidMass, unsigned int* centroidCount)
{
int const af_id = blockIdx.x;
int const cluster_id = blockIdx.y;
int const filter_id = threadIdx.x;

int local_mass = 0;
int local_count = 0;

if (af_id == 0)
{
int idx0 = filter_id*64 + cluster_id;

for (int i=0; i<gridDim.x; i++)
{
int idxother = i * gridDim.y*blockDim.x + idx0;

local_mass += centroidMass[idxother];
local_count += centroidCount[idxother];
}

centroidMass[idx0] = local_mass;
centroidCount[idx0] = local_count;
}
}
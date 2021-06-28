#include "includes.h"
__global__ void findCentroidsAtomicFreeLocal(int afLocal, int* responses, int nPixels, int* cluster, int* centroidMass, unsigned int* centroidCount)
{
int const af_id = blockIdx.x;
int const cluster_id = blockIdx.y;
int const filter_id = threadIdx.x;
int* filter_responses = &responses[filter_id*nPixels];

int local_responses = 0;
int local_count = 0;

int pixel_start = af_id*afLocal;
int pixel_end = (af_id+1)*afLocal;

pixel_end = pixel_end>nPixels?nPixels:pixel_end;

for (int i=pixel_start; i<pixel_end; i++)
{
if (cluster[i] == cluster_id)
{
local_responses += filter_responses[i];
local_count++;
}
}

int idx = af_id * gridDim.y*blockDim.x + filter_id*32 + cluster_id;
centroidMass[idx] = local_responses;
centroidCount[idx] = local_count;
}
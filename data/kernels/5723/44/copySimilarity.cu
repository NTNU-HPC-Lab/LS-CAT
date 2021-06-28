#include "includes.h"
__global__ void copySimilarity(float* similarities, int active_slices, int slices, int* activeMask, int target, int source)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i >= active_slices)
return;
int slice = activeMask[i];
similarities[target*slices + slice] = similarities[source*slices + slice];
}
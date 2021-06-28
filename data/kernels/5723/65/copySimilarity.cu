#include "includes.h"
__global__ void copySimilarity(float* similarities, int active_patches, int patches, int* activeMask, int target, int source)
{
int i = threadIdx.x + blockIdx.x * blockDim.x;
if (i >= active_patches)
return;
int patch = activeMask[i];
similarities[target*patches + patch] = similarities[source*patches + patch];
}
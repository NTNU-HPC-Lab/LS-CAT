#include "includes.h"
__global__ void findRadixOffsets(uint2* keys, uint* counters, uint* blockOffsets, uint startbit, uint numElements, uint totalBlocks)
{
__shared__ uint  sStartPointers[16];
extern __shared__ uint sRadix1[];

uint groupId = blockIdx.x;
uint localId = threadIdx.x;
uint groupSize = blockDim.x;

uint2 radix2;
radix2 = keys[threadIdx.x + (blockIdx.x * blockDim.x)];

sRadix1[2 * localId]     = (radix2.x >> startbit) & 0xF;
sRadix1[2 * localId + 1] = (radix2.y >> startbit) & 0xF;

// Finds the position where the sRadix1 entries differ and stores start
// index for each radix.
if(localId < 16)
{
sStartPointers[localId] = 0;
}
__syncthreads();

if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
{
sStartPointers[sRadix1[localId]] = localId;
}
if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1])
{
sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
}
__syncthreads();

if(localId < 16)
{
blockOffsets[groupId*16 + localId] = sStartPointers[localId];
}
__syncthreads();

// Compute the sizes of each block.
if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) )
{
sStartPointers[sRadix1[localId - 1]] =
localId - sStartPointers[sRadix1[localId - 1]];
}
if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] )
{
sStartPointers[sRadix1[localId + groupSize - 1]] =
localId + groupSize - sStartPointers[sRadix1[localId +
groupSize - 1]];
}

if(localId == groupSize - 1)
{
sStartPointers[sRadix1[2 * groupSize - 1]] =
2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
}
__syncthreads();

if(localId < 16)
{
counters[localId * totalBlocks + groupId] = sStartPointers[localId];
}
}
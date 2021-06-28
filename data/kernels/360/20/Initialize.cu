#include "includes.h"
__global__ void Initialize(int size, unsigned int *randoms, int *bestSeen, int *origin, int *mis, int *incomplete) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < size)
{
// Taustep is performed with S1=13, S2=19, S3=12, and M=UINT_MAX coded into kernel
unsigned int z = randoms[idx];
unsigned int b = (((z << 13) ^ z) >> 19);
z = (((z & UINT_MAX) << 12) ^ b);

// Set the origin to be self
origin[idx] = idx;

// Set the bestSeen value to be either random from 0-1000000 or 1000001 if in MIS
int status = mis[idx];
int value = 0;
if (status == 1)
value = 1000001;

bestSeen[idx] = (mis[idx] == -1) ? (z % 1000000) : value;

// Write out new random value for seeding
randoms[idx] = z;
}

// Reset incomplete value
if (idx == 0)
incomplete[0] = 0;
}
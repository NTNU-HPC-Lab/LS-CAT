#include "includes.h"



__global__ void rnd_init(size_t _count, curandState * _state)
{
int index = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y + (blockIdx.x * blockDim.x * blockDim.y * blockDim.z);
if (index >= _count)
{
return;
}
//Each thread gets same seed, a different sequence number, no offset
curand_init(1984, index, 0, &_state[index]);
}
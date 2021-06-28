#include "includes.h"
__global__ void cudaSgatherI2I_kernel( const int* keys, const int* indicesX, const int* indicesY, const int* indicesK, int* outX, int* outY, int* outK, unsigned int nbElements)
{
const int index = (threadIdx.x & 0x1f) + blockIdx.x*blockDim.x;

if(index < nbElements)
{
const int key = keys[index];
printf("keys[%d]=%d indicesX[%d]:%d  ", index, key, index, indicesX[index] );
outX[index] = indicesX[key];
outY[index] = indicesY[key];
outK[index] = indicesK[key];
}
}
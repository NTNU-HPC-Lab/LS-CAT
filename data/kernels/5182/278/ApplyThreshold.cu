#include "includes.h"
__global__ void ApplyThreshold( float* probabilitiesInputs, float* binaryOutput, float* probability, int count ) {
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;

if (id < count)
{
if (probabilitiesInputs[id] < probability[0])
{
binaryOutput[id] = 0.0f;
}
else
{
binaryOutput[id] = 1.0f;
}
}
}
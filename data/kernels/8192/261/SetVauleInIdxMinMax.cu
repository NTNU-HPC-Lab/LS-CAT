#include "includes.h"
__global__ void SetVauleInIdxMinMax( float* vector, int id_min, int id_max, float value)
{
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
if (id >= id_min && id <= id_max)
vector[id] = value;
}
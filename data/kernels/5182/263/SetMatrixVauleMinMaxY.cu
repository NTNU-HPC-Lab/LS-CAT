#include "includes.h"
__global__ void SetMatrixVauleMinMaxY( float* matrix, int cols, int size, int id_min, int id_max, float value)
{
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
int id_row = id / cols;
if (id_row >= id_min && id_row <= id_max && id < size)
matrix[id] = value;
}
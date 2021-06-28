#include "includes.h"
__global__ void SetMatrixVauleMinMaxX( float* matrix, int cols, int size, int id_min, int id_max, float value)
{
int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
int id_column = id%cols;
if (id_column >= id_min && id_column <= id_max && id < size)
matrix[id] = value;
}
#include "includes.h"
__global__ void mult_complex_eff_kernal(float* data, const float* src_data, const int nx, const int nxy, const int size)
{
int idx = threadIdx.z*nxy + threadIdx.y*nx + threadIdx.x;

data[idx] *= src_data[idx];
data[size-idx-1] *= src_data[size-idx-1];
}
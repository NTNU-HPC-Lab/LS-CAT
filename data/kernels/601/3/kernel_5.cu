#include "includes.h"
__global__ void kernel_5(int *new_data, int *data)
{
int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

if (_tid_ >= 10000000) return;

int idx_2 = (_tid_ / 2) % 500;

new_data[_tid_] = (data[_tid_] + idx_2) % 13377;
}
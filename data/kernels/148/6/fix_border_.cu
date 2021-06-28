#include "includes.h"


#define TB 128
#define GS(x) (((x) - 1) / TB + 1)

__global__ void fix_border_(float *input, int pad_size, int side, int size3, int size23)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < size23) {
int x = id % size3;
int y = id / size3;
if (side == 0 && x < pad_size) {
input[id] = input[y * size3 + pad_size];
} else if (side == 1 && x > size3 - pad_size - 1) {
input[id] = input[y * size3 + size3 - pad_size - 1];
}
}
}
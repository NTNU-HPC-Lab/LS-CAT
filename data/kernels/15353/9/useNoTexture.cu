#include "includes.h"
__global__ void useNoTexture(float* pin, float* pout, int len)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

auto a = pin[0 + len * (i + c_size.x * (j + k * c_size.y))];
auto b = pin[1 + len * (i + c_size.x * (j + k * c_size.y))];
auto c = pin[2 + len * (i + c_size.x * (j + k * c_size.y))];

pout[i + c_size.x * (j + k * c_size.y)] = sqrtf(powf(a, 2) + powf(b, 2) + powf(c, 2));

}
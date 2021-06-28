#include "includes.h"
__global__ void backProp2(float* layer1, float* dsyn2, float* label, float* out)
{
int j = blockDim.x*blockIdx.x + threadIdx.x;
int k = blockDim.y*blockIdx.y + threadIdx.y;
float delta = (label[k] - out[k]) * (out[k]*(1.0-out[k]));
dsyn2[j*10 + k] += delta * layer1[j] / (60000.0/10.0);
}
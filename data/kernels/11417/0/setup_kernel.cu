#include "includes.h"



#define PI 3.14159265359

#define DEG_TO_RAD (PI / 180.0)

typedef unsigned char byte;

__global__ void setup_kernel(curandState* state, unsigned long seed)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
curand_init(seed, id, 0, &state[id]);
}
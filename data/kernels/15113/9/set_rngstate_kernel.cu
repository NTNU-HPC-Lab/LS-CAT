#include "includes.h"
__global__ void set_rngstate_kernel(curandStateMtgp32 *state, mtgp32_kernel_params *kernel)
{
state[threadIdx.x].k = kernel;
}
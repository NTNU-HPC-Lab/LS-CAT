#include "includes.h"
__global__ void boundaryCondition_k(float* payoff, size_t spotSize, float strike) {
size_t state_idx = threadIdx.x;
payoff[spotSize - 1 + state_idx * spotSize] = 2 * strike;
payoff[0 + state_idx * spotSize] = 0.0;
}
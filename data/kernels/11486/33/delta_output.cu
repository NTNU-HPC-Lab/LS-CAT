#include "includes.h"
__global__ void delta_output ( const float * primed_sum, const float * ideal, const float * actual, float * delta, unsigned int index )
{
// x is the output neuron/node count (e.g., length of actual & ideal)
int x = blockIdx.x * blockDim.x + threadIdx.x;

// Calculate the Negative Error: -(Actual - Ideal)
float neg_error = __fmul_rz(-1,(actual[x] - ideal[x]));

// -E * σ'(Σ(O[i])
delta[x+index] = __fmul_rz( neg_error, primed_sum[x+index] );
}
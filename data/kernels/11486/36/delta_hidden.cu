#include "includes.h"
__global__ void delta_hidden ( float * prime_ji, float * delta_i )
{
// X grid is size_i
int x = blockIdx.x * blockDim.x + threadIdx.x;
// δ[i] = f'( Σ[ji]) * Σ(w[ik] * δ[k])
// NOTE: delta_i ALREADY contains `Σ(w[ik] * δ[k])`
float rhs = delta_i[x];
// δ[i] = σ'( Σ[ji]) * Σ(w[ik] * δ[k])
delta_i[x] = __fmul_rz( prime_ji[x], rhs );
}
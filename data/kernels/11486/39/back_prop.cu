#include "includes.h"
__global__ void back_prop ( float * weight, float * gradient, float * update, float alpha, float epsilon )
{
// X Grid iterates weight, gradient and update (all same size)
int x = blockIdx.x * blockDim.x + threadIdx.x;
// ε * ( ∂E / ∂W[ik] )
float lhs = __fmul_rz( epsilon, gradient[x] );
// α * ( Δw(t-1) )
float rhs = __fmul_rz( alpha, update[x] );
// Δw(t) = ε * ( ∂E / ∂W[i] ) + α * ( Δw(t-1) )
float d_w = __fadd_rz( lhs, rhs );

//printf("Δw(t): %f W[i]: %f W[i]+Δw(t): %f Δw(t-1): %f\n",d_w,weight[x],__fadd_rz(weight[x],d_w),update[x]);

// Update weight: W[i] = W[i] + Δw(t)
weight[x] = __fadd_rz( weight[x], d_w );
// Set `Δw(t-1) = Δw(t)`
update[x] = d_w;
}
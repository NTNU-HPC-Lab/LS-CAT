#include "includes.h"
__global__ void uint_arithm(float* A, float* B, float* C, unsigned int u1, unsigned int u2)
{
// device function call (warn if unsupported)
unsigned int _umin = umin ( u1, u2 );
// device function call (warn if unsupported)
unsigned int _umax = umax ( u1, u2 );
// device function call (warn if unsupported)
unsigned int _umin_global = ::umin ( u1, u2 );
// device function call (warn if unsupported)
unsigned int _umax_global = ::umax(u1, u2);
if (_umin != _umin_global) return;
if (_umax != _umax_global) return;
int i = threadIdx.x;
A[i] = i + _umin;
B[i] = i + _umax;
C[i] = A[i] + B[i];
}
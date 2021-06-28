#include "includes.h"

#define tam 1.0
#define dx 0.00001
#define dt 0.000001
#define T 0.01
#define kappa 0.000045





__global__ void Inicializacao( double *uprev, const int n ) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
double x = idx * dx;
if( idx < n + 1 ) {
if( x <= 0.5 ) {
uprev[ idx ] = 200 * x;
}
else {
uprev[ idx ] = 200 * ( 1. - x );
}
}
}
#include "includes.h"

#define tam 1.0
#define dx 0.00001
#define dt 0.000001
#define T 0.01
#define kappa 0.000045





__global__ void Atualiza( double *u, double *u_prev, const int n ) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if( idx == 0 ) {
u[ 0 ] = u[ n ] = 0.; /* forca condicao de contorno */
}
else if( idx < n ) {
u[ idx ] = u_prev[ idx ] + kappa * dt / ( dx * dx ) * ( u_prev[ idx - 1 ] - 2 * u_prev[ idx ] + u_prev[ idx + 1 ] );
}
}
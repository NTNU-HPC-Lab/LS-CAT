#include "includes.h"
__global__ void stencil_1d(int n, double *in, double *out)
{
/* calculate global index in the array */
int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

/* return if my global index is larger than the array size */
if( globalIndex >= n ) return;

/* code to handle the boundary conditions */
if( globalIndex < RADIUS || globalIndex >= (n - RADIUS) )
{
out[globalIndex] = (double) globalIndex * ( (double)RADIUS*2 + 1) ;
return;
} /* end if */

double result = 0.0;

for( int i = globalIndex-(RADIUS); i <= globalIndex+(RADIUS); i++ )
{
result += in[i];
}

out[globalIndex] = result;
return;

}
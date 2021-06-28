#include "includes.h"
__global__ void stencil_1d(int n, double *in, double *out)
{
/* allocate shared memory */
__shared__ double temp[THREADS_PER_BLOCK + 2*(RADIUS)];

/* calculate global index in the array */
int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
int localIndex = threadIdx.x + RADIUS;

/* return if my global index is larger than the array size */
if( globalIndex >= n ) return;

/* read input elements into shared memory */
temp[localIndex] = in[globalIndex];

/* code to handle the halos.  need to make sure we don't walk off the end
of the array */
if( threadIdx.x < RADIUS && globalIndex >= RADIUS )
{
temp[localIndex - RADIUS] = in[globalIndex - RADIUS];
} /* end if */

if( threadIdx.x < RADIUS && globalIndex < (n - RADIUS) )
{
temp[localIndex + THREADS_PER_BLOCK] = in[globalIndex + THREADS_PER_BLOCK];
} /* end if */

/* code to handle the boundary conditions */
if( globalIndex < RADIUS || globalIndex >= (n - RADIUS) )
{
out[globalIndex] = (double) globalIndex * ( (double)RADIUS*2 + 1) ;
return;
} /* end if */

double result = 0.0;

for( int i = -(RADIUS); i <= (RADIUS); i++ )
{
result += temp[localIndex + i];
} /* end for */

out[globalIndex] = result;
return;

}
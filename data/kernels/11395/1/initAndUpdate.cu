#include "includes.h"
__global__ void initAndUpdate( float *D_oldVal, float *D_currVal, int tpoints, int nsteps )
{
int j = blockDim.x * blockIdx.x + threadIdx.x;
if ( j < tpoints )
{
j += 1;
/* Calculate initial values based on sine curve */
/* Initialize old values array */
float x = ( float )( j - 1 ) / ( tpoints - 1 );
D_oldVal[j] = D_currVal[j] = sin ( 6.2831853f * x );
int i;
/* global endpoints */
if ( ( j == 1 ) || ( j  == tpoints ) )
{
D_currVal[j] = 0.0;
}
else
{
/* Update values for each time step */
for ( i = 1; i <= nsteps; i++ )
{
/* Update old values with new values */
float newVal = ( 2.0 * D_currVal[j] ) - D_oldVal[j] + ( 0.09f * ( -2.0 ) * D_currVal[j] );
D_oldVal[j] = D_currVal[j];
D_currVal[j] = newVal;
}
}
}
}
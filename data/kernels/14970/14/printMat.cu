#include "includes.h"
__global__ void printMat( const double *A, int size )
{
if( threadIdx.x == 0 && blockIdx.x == 0 )
for( int i = 0; i < size; i++ )
printf("A[%d] = %f\n",i,A[i]);
return;
} /* end printMat */
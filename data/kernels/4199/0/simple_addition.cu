#include "includes.h"
/*
Vector addition with a single thread for each addition
*/



/*
Vector addition with thread mapping and thread accessing its neighbor parallely
*/

//slower than simpler


/*
Matrix Matrix multiplication with a single thread for each row
*/


/*
Matrix Matrix multiplication with a single thread for each result element
*/


/*
Matrix Vector multiplication with a block with 4 threads per block, shared block mem and parallel reduce
*/


__global__ void simple_addition(int *a, int *b,int *c,int len)
{
int tid=threadIdx.x +blockIdx.x*blockDim.x ;
//while (tid<len)
c[tid]=a[tid]+b[tid];
//printf("I am block: %d with tid: %d Result: %d \n",blockIdx.x,tid,c[tid]);

}
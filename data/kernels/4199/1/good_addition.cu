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


__global__ void good_addition(int *a, int *b, int *c, int len)
{
int tid= threadIdx.x + blockIdx.x * blockDim.x;
const int thread_count= blockDim.x*gridDim.x;
int step = len/thread_count;

int start_index = tid*step;
int end_index= (tid+1)* step;
if (tid==thread_count-1) end_index=len;
//printf("Step is %d\n",step);
while(start_index< end_index)
{
c[start_index]=a[start_index]+b[start_index];

//printf("I am block: %d with tid: %d Result %d \n",blockIdx.x,tid,c[tid]);
start_index +=1;
}
}
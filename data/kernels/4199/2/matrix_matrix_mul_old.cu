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


__global__ void matrix_matrix_mul_old(int *a, int *b, int *c, int n_row, int n_col, int n_comm)

{
int tid= threadIdx.x + blockIdx.x * blockDim.x;
int temp=0;
while(tid<n_row)
{
for (int k=0;k<n_col;k++)
{
temp=0;
for(int j=0;j<n_comm;j++)
{
temp+= a[n_comm*tid+j]* b[j*n_col+k];
}
c[tid*n_col+k]=temp;
}
tid+=blockDim.x * gridDim.x;

}
}
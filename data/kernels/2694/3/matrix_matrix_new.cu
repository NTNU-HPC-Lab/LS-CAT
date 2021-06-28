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


__global__ void matrix_matrix_new(int *a, int *b, int *c, int n_row, int n_col, int n_comm)
{
int tid= threadIdx.x + blockIdx.x *  blockDim.x;
int temp=0;
while(tid<n_row*n_col)
{
// find the row index of A
int i=tid / n_col;
// find the column index of B
int j=tid % n_col;
// multiply the row and column
temp=0;
for(int k=0;k<n_comm;k++)
{
temp+= a[i*n_comm+k]*b[j+k*n_col];
}
c[tid]=temp;
tid+= blockDim.x * gridDim.x;
}
}
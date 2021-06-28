#include "includes.h"
__global__ void kDot(const int nThreads, const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_columns ){
/*  Computes the product of two matrices: m1 x m2.
Inputs:
m1: array, left matrix of size m1_rows x m1_columns
m2: array, right matrix of size m1_columns x m2_columns (the number of rows in the right matrix
must be equal to the number of the columns in the left one)
output: array, the results of the computation are to be stored here:
m1 * m2, product of two arrays m1 and m2, a matrix of size m1_rows x m2_columns
m1_rows: int, number of rows in the left matrix m1
m1_columns: int, number of columns in the left matrix m1
m2_columns: int, number of columns in the right matrix m2
*/

for (int i = blockIdx.x * blockDim.x + threadIdx.x;
i < nThreads;
i += blockDim.x * gridDim.x)
{
int r = (int)i / m2_columns;
int c = i % m2_columns;
float t_output = 0.f;

for( int k = 0; k < m1_columns; ++k ) {
t_output += m1[ r * m1_columns + k ] * m2[ k * m2_columns + c ];
}

output[i] = t_output;
}
}
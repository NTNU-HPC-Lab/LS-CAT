#include "includes.h"
__global__ void chol_kernel_optimized_div(float * U, int k, int stride) {
//With stride...

//General thread id
int tx = blockIdx.x * blockDim.x + threadIdx.x;

//Iterators
unsigned int j;
unsigned int num_rows = MATRIX_SIZE;

//Only let one thread do this
if (tx == 0) {
// Take the square root of the diagonal element
U[k * num_rows + k] = sqrt(U[k * num_rows + k]);
//Don't bother doing check...live life on the edge!
}

//Each thread does some part of j
//Stide in units of 'stride'
//Thread 0 does 0, 16, 32
//Thread 1 does 1, 17, 33
//..etc.
int offset = (k + 1); //From original loop
int jstart = threadIdx.x + offset;
int jstep = stride;

//Only continue if in bounds?
//Top limit on i for whole (original) loop
int jtop = num_rows - 1;
//Bottom limit on i for whole (original) loop
int jbottom = (k + 1);

//Do work for this i iteration
//Division step
//Only let one thread block do this
if (blockIdx.x == 0) {
for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
U[k * num_rows + j] /= U[k * num_rows + k]; // Division step
}
}
}
#include "includes.h"
__global__ void chol_kernel_optimized(float * U, int k, int stride) {
//With stride...

//Iterators
unsigned int j;
unsigned int num_rows = MATRIX_SIZE;


//This call acts as a single K iteration
//Each block does a single i iteration
//Need to consider offset,
int i = blockIdx.x + (k + 1);
//Each thread does some part of j
//Stide in units of 'stride'
//Thread 0 does 0, 16, 32
//Thread 1 does 1, 17, 33
//..etc.
int offset = i; //From original loop
int jstart = threadIdx.x + offset;
int jstep = stride;

//Only continue if in bounds?
//Top limit on i for whole (original) loop
int jtop = num_rows - 1;
//Bottom limit on i for whole (original) loop
int jbottom = i;

//Do work for this i iteration
//Want to stride across
for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
}
}
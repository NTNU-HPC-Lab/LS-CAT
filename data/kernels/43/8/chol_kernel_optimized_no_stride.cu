#include "includes.h"
__global__ void chol_kernel_optimized_no_stride(float * U, int k, int stride) {
//Iterators
unsigned int j;
unsigned int num_rows = MATRIX_SIZE;

//TODO USE STRIDE

//This call acts as a single K iteration
//Each block does a single i iteration
//Need to consider offset,
int i = blockIdx.x + (k + 1);
//Each thread does some part of j
//Split j based on stride and thread index
//Index 0 is j= 0-15
//Index 1 is j=16-31
int offset = i;
int jstart = (threadIdx.x * stride) + offset;
int jend = jstart + (stride - 1);

//Only continue if in bounds?
//Top limit on i for whole (original) loop
int jtop = num_rows - 1;
//Bottom limit on i for whole (original) loop
int jbottom = i;
//Check boundaries, else do nothing
if (!((jstart >= jbottom) && (jend <= jtop))) {
return; //This thread does nothing now
}

//Do work  for this i iteration
//Want to stride across
for (j = jstart; j <= jend; j++) {
U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
}
}
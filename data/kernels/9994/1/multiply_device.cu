#include "includes.h"
__global__ void multiply_device (double *d_a, double *d_b,int dim) {

//Declaration of required variables.
double a, b, sum;

//Retrive the thread and block specific information.
int i = threadIdx.x,j,k;

// Begine Matrix Computation.
for (j = blockIdx.x; j < dim; j += gridDim.x) {
sum = 0;
for(k=0; k<dim; k++) {
a =d_a[k *dim+i];
b =d_a[k*dim+j];
sum  = sum + a * b;
}
d_b[ i * dim + j ] = sum;
}
}
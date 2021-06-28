#include "includes.h"


__global__ void matrix_multiply_kernel(double *matrix, double *vector_in, double *vector_out, long dim_mn){
double out;
long i, j;
i = threadIdx.x + blockIdx.x * blockDim.x;
if (i<dim_mn){
out = 0.;
for (j=0; j<dim_mn; j++){
out += matrix[i*dim_mn+j] * vector_in[j];
}
vector_out[i] = out;
}
}
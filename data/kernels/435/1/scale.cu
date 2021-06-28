#include "includes.h"
/**
* C file for parallel QR factorization program usign CUDA
* See header for more infos.
*
* 2016 Marco Tieghi - marco01.tieghi@student.unife.it
*
*/



#define THREADS_PER_BLOCK 512   //I'll use 512 threads for each block (as required in the assignment)

__global__ void scale(double *d, int m, int ld, double *s) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx < m) {
d[idx*ld] = d[idx*ld] / sqrt(*s);    //Applying scale
}
}
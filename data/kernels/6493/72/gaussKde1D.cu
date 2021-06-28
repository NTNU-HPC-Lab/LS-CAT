#include "includes.h"
__global__ void gaussKde1D ( const int dim, const int nd, const int nb, const int Indx, const float *hh, const float *a, const float *b, float *pdf ) {
int i = threadIdx.x + blockDim.x * blockIdx.x;
int j = threadIdx.y + blockDim.y * blockIdx.y;
int ij = i + j * nb;
float h;
if ( i < nb && j < nd ) {
h = hh[Indx];
pdf[ij] = expf ( - powf ( a[Indx+j*dim] - b[Indx+i*dim], 2. ) / 2. / powf ( h, 2 ) ) / h / powf ( 2 * PI, 0.5 );
}
}
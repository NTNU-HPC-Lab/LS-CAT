#include "includes.h"
__global__ void floyd1DKernel(int * M, const int nverts, const int k){
int ii = blockIdx.x * blockDim.x + threadIdx.x;    // indice filas, coincide con ij
int i = ii/nverts;
int j = ii - i * nverts;

if(i < nverts && j < nverts){
int kj = (k*nverts) + j;
// printf("TID = %u \n\tI = %u => \tM[%u] = %u \n \tK = %u => \tM[%u] = %u  \n", ii, i, ii, M[ii], k, kj, M[kj]);
if (i!=j && i!=k && j!=k) {
int ik = (i*nverts) + k;
// int kj = (k*nverts) + j;
M[ii] = min(M[ik] + M[kj], M[ii]);
}
}
}
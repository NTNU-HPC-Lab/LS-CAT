#include "includes.h"
__global__ void floyd2DKernel(int * M, const int nverts, const int k){
int jj = blockIdx.x * blockDim.x + threadIdx.x; // indice filas
int ii = blockIdx.y * blockDim.y + threadIdx.y; // indice columnas
int tid = (ii * nverts) + jj;
int i = tid/nverts;
int j = tid - i * nverts;
//printf ("Fila %u, Columna %u => Thread id %d.\n", i, j, tid);

if(i < nverts && j < nverts){
if (i!=j && i!=k && j!=k) {
int ik = (i*nverts) + k;
int kj = (k*nverts) + j;
int ij = (i*nverts) + j;
int aux = M[ik]+M[kj];

int vikj = min(aux, M[ij]);
M[ij] = vikj;
}
}
}
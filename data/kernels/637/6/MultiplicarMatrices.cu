#include "includes.h"
__global__ void MultiplicarMatrices(float *m1, float *m2, float *mr, int columna1, int fila1, int columna2, int fila2)
{
int fila_r = blockIdx.y*blockDim.y+threadIdx.y;
int columna_r = blockIdx.x*blockDim.x+threadIdx.x;
float tmp_mult = 0;

if ((fila_r < fila2) && (columna_r < columna1)) {
for (int i = 0; i < fila2 ; i++) {
tmp_mult += m1[i+columna1*fila_r]*m2[i*columna2+columna_r];
}

mr[fila_r*columna2+columna_r]= tmp_mult;
}
}
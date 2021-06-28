#include "includes.h"
__global__ void add(float *A, float *C)
{
int columna = threadIdx.x;
//indice de las filas
int fila = threadIdx.y;
//indice lineal
int Id = columna + fila * blockDim.x;

int id1 = (columna - 1) + fila * blockDim.x;
int id2 = (columna + 1) + fila * blockDim.x;
int id3 = columna + (fila - 1) * blockDim.x;
int id4 = columna + (fila + 1) * blockDim.x;

if ((fila > 0 && fila < N - 1) && (columna > 0 && columna < N - 1)) {

C[Id] = A[id1] + A[id2] + A[id3] + A[id4];
}
else
{
C[Id] = A[Id];
}
}
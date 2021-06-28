#include "includes.h"
__global__ void suma(float *A, float *B, float *C)
{
//indice de las columnas
int columna = threadIdx.x;
//indice de las filas
int fila = threadIdx.y;
//indice lineal
int Id = columna + fila * blockDim.x;
//sumamos cada elemento
C[Id] = A[Id] + B[Id];
}
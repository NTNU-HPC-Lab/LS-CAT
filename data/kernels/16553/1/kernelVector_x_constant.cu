#include "includes.h"
__global__ void kernelVector_x_constant( float* arr, int n, int k )
{
//Obtengo el indice del hilo fisico
int idx = blockIdx.x * blockDim.x + threadIdx.x;

//Mientras el hilo sea valido para la operaci�n
if( idx<n )
{
//Multiplico el elemento por la constante
arr[ idx ] = arr[ idx ] * k;
}
}
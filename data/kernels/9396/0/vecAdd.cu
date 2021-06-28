#include "includes.h"
/**
* Maestría en Ciencias - Mención Informática
* -------------------------------------------
* Escriba un programa CUDA que calcule C = n*A + B, en donde A, B, C son vectores
* y n una constante escalar.
*
* Adaptado de https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/
*
* Presentado por:
* Zuñiga Rojas, Gabriela
* Soncco Pimentel, Braulio
*/

cudaEvent_t start, stop;
float elapsedTime;

const int k = 5;

// CUDA kernel. Each thread takes care of one element of c

__global__ void vecAdd(double *a, double *b, double *c, int n, int k)
{

// Get our global thread ID
int id = blockIdx.x*blockDim.x+threadIdx.x;

// Make sure we do not go out of bounds
if (id < n)
c[id] = k * a[id] + b[id];

}
#include "includes.h"
__global__ void elevar_al_cuadrado(float * d_salida, float * d_entrada){
int idx = threadIdx.x;
float f = d_entrada[idx];
d_salida[idx] = f*f;
}
#include "includes.h"
__global__ void cudahello(){
int thread = threadIdx.x;
int block = blockIdx.x;
printf("Hola Mundo! Soy el hilo %d del bloque %d\n", thread, block);
}
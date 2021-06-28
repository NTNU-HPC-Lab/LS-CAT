#include "includes.h"
/**
* calculate pi
*/
// For the CUDA runtime routines (prefixed with "cuda_")
//Tiempo

#define NUMTHREADS 10240
#define ITERATIONS 1e12

/**
* CUDA Kernel Device code
*
*/
/*****************************************************************************/



/******************************************************************************
* Host main routine
*/
__global__ void calculatePi(double *piTotal, long int iterations, int totalThreads)
{   long int initialIteration, endIteration;
long int i = 0;
double piPartial;

//TamanioBloque*IdBloque + IdHilo
int index = (blockDim.x * blockIdx.x) + threadIdx.x;

initialIteration = (iterations/totalThreads) * index;
endIteration = initialIteration + (iterations/totalThreads) - 1;

i = initialIteration;
piPartial = 0;

do{
piPartial = piPartial + (double)(4.0 / ((i*2)+1));
i++;
piPartial = piPartial - (double)(4.0 / ((i*2)+1));
i++;
}while(i < endIteration);

piTotal[index] = piPartial;

__syncthreads();
if(index == 0){
for(i = 1; i < totalThreads; i++)
piTotal[0] = piTotal[0] + piTotal[i];
}
}
#include "includes.h"
__global__ void calculatePi(double *piTotal, long int iterations, int totalThreads)
{   long int initIteration, endIteration;
long int i = 0;
double piPartial;

int index = (blockDim.x * blockIdx.x) + threadIdx.x;

initIteration = (iterations/totalThreads) * index;
endIteration = initIteration + (iterations/totalThreads) - 1;

i = initIteration;
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
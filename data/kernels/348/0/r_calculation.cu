#include "includes.h"

char* concat(char *s1, char *s2);





__global__ void r_calculation(float* a , int * indeces , float* b , float* x,float * r  ,int size)
{
int index = blockDim.x * blockIdx.x + threadIdx.x ;

if (index < size)
{
float sum = 0 ;

for (int i = 0 ; i<3 ; i++)
{
sum += a[3*index  + i] * x[indeces[3*index + i]] ;
}

r[index] = b[index] - sum ;
}

}
#include "includes.h"

#define MINVAL 1e-7

__global__ void Permute(double* Dev_Mtr, int* i, int* k, int* Dev_size)
{
int index=blockDim.x*blockIdx.x+threadIdx.x;

if(index<*Dev_size)
{
double tmp=Dev_Mtr[index*(*Dev_size)+(*i)];
Dev_Mtr[index*(*Dev_size)+(*i)]=Dev_Mtr[index*(*Dev_size)+(*k)];
Dev_Mtr[index*(*Dev_size)+(*k)]=tmp;
}

}
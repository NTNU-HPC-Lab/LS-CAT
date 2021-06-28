#include "includes.h"

#define MINVAL 1e-7

__global__ void Gaus(double* Mtr, int Size, int i)
{
int index=blockIdx.x*blockDim.x+threadIdx.x;
if(index>i && index< Size)
{
double particial = -Mtr[i*Size+index]/Mtr[i*Size+i];

for(int z=i; z<Size; z++)
{
Mtr[z*Size+index]=Mtr[z*Size+index]+Mtr[z*Size+i]*particial;
}

}
}
#include "includes.h"
extern "C"
__global__ void bubble(unsigned int length, unsigned int parity, float* tab)
{

int index = 2* (threadIdx.x + blockDim.x * blockIdx.x);
int  leftElementID = index + parity;
int rightElementID = index + parity + 1;

float l, r;
if (rightElementID < length)
{
l = tab[  leftElementID ];
r = tab[ rightElementID ];
if ( r < l )
{
tab[  leftElementID ] = r;
tab[ rightElementID ] = l;
}
}


}
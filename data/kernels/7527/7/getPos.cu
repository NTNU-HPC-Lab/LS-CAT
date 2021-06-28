#include "includes.h"
__global__ void getPos(int *d_scanArray , int d_numberOfElements,int *d_lastPos)
{
*d_lastPos = d_scanArray[d_numberOfElements -1];
}
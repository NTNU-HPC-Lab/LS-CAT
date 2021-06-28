#include "includes.h"
__global__ void ThirdAngle(int *a1, int *a2, int *a3)
{
*a3 = (180-*a1-*a2);
}
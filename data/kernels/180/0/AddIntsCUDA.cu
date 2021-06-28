#include "includes.h"


__global__ void AddIntsCUDA(int *a, int *b) //Kernel Definition
{
*a = *a + *b;
}
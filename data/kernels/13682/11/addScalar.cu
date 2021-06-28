#include "includes.h"
__global__ void addScalar(int a, int b, int* ptrC)
{
*ptrC = a + b; // Hyp: 1 seul thread

// debug
printf("[GPU] %d + %d = %d", a, b, *ptrC);
}
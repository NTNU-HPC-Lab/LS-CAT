#include "includes.h"
/*
1. Memory Copy Cost   One-Step
2. Straggler: Ring-based
**/
using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

__global__ void helloFromGPU(void)
{
printf("Hello from GPU\n");
}
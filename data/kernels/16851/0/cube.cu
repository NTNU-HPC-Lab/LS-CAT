#include "includes.h"



using namespace std;


__global__ void cube(long *deviceOutput, long *deviceInput)
{
int idx = threadIdx.x;
long f = deviceInput[idx];
deviceOutput[idx] = f * f * f;
}
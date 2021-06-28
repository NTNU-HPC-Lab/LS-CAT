#include "includes.h"
__device__ int power_modulo_fast(long a, long b, long m)
{
long i;
long result = 1;
long  x = a%m;

for (i=1; i<=b; i<<=1)
{
x %= m;
if ((b&i) != 0)
{
result *= x;
result %= m;
}
x *= x;
}

return result;
}
__device__ float generate( curandState* globalState, int ind )
{
//int ind = threadIdx.x;
curandState localState = globalState[ind];
float RANDOM = curand_uniform( &localState );
globalState[ind] = localState;
return RANDOM;
}
__global__ void kernel(int p, bool* prime, curandState* globalState)
{
int i = blockIdx.x;
int a = 0;
while(*prime && i < PRECISION)
{
a = (generate(globalState, i%BLOCKS_NUBMER) * (p-2))+1;
if(power_modulo_fast(a, p-1, p) == 1)
{
i += BLOCKS_NUBMER;
}
else
{
*prime = false;
}
}

}
#include "includes.h"
__global__ static void segmentSieve(char *primes, uint64_t max) {
int index = blockIdx.x * blockDim.x + threadIdx.x;
if(index>0){
const uint64_t maxRoot = sqrt((double)max);
int low = maxRoot*index;
int high = low + maxRoot;
if(high > max) high = max;
for (int i = 2; i < maxRoot; i++){ //sqrt(n)lglg(sqrt(n))
if(primes[i]==0){
int loLim = (low / i) * i;
if (loLim < low)
loLim += i;
for (int j=loLim; j<high; j+=i)
primes[j] = 1;
}

}
}
}
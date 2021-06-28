#include "includes.h"
__global__ static void sieveOfEratosthenes(char *primes, uint64_t max) {
primes[0] = 1; // value of 1 means the number is NOT prime
primes[1] = 1; // numbers "0" and "1" are not prime numbers
int index = blockIdx.x * blockDim.x + threadIdx.x;
const uint64_t maxRoot = sqrt((double)max);
// make sure index won't go out of bounds,
// also don't execute it on index 1
if (index <= maxRoot && primes[index] == 0 && index > 1 ){
// mark off the composite numbers
for (int j = index * index; j < max; j += index){
primes[j] = 1;
}

}
}
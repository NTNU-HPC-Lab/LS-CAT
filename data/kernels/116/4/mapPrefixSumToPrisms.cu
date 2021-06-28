#include "includes.h"
__global__ void mapPrefixSumToPrisms( const unsigned numberOfPrisms, const unsigned raysPerSample, const unsigned reflectionSlices, const unsigned* raysPerPrism, const unsigned* prefixSum, unsigned *indicesOfPrisms, unsigned *numberOfReflections ){

int id = threadIdx.x + (blockIdx.x * blockDim.x);
// break if we have too many threads (this is likely)
if(id >= numberOfPrisms*reflectionSlices) return;

const unsigned count            = raysPerPrism[id];
const unsigned startingPosition = prefixSum[id];
const unsigned reflection_i     = id / numberOfPrisms;
const unsigned prism_i          = id % numberOfPrisms;

for(unsigned i=0; i < count ; ++i){
indicesOfPrisms[startingPosition + i] = prism_i;
numberOfReflections[startingPosition + i] = reflection_i;
}
}
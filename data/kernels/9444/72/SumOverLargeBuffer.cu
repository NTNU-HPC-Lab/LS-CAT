#include "includes.h"
__global__ void SumOverLargeBuffer( float* buffer, int spread, int size ){

int offset = CUDASTDOFFSET;
float value1 = buffer[offset];
float value2 = buffer[offset+spread];

if( offset+spread < size )
buffer[offset] = value1+value2;

}
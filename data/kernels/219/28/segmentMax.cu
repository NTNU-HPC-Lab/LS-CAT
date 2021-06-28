#include "includes.h"
__device__ float getAbsMax(float * d_vec, const int length)
{
int jj=0;
float segmentMax = 0;

for (jj=0; jj<length; jj++) {
if ( segmentMax < abs(d_vec[jj]) ) segmentMax = abs(d_vec[jj]);
}

return segmentMax;
}
__global__ void segmentMax(float* d_vec, float *segmentMaxes, const int length, const int HighLength, const int HighSegmentLength, const int threadsHigh, const int LowSegmentLength)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int startIndex, SegmentLength;

if ( (xIndex*HighSegmentLength > HighLength) & ( (HighLength + (xIndex-threadsHigh+1)*LowSegmentLength) < length ) ){
startIndex = HighLength + (xIndex-threadsHigh)*LowSegmentLength;
SegmentLength = LowSegmentLength;
}
else {
startIndex = xIndex*HighSegmentLength;
SegmentLength = HighSegmentLength;
}
segmentMaxes[xIndex] = getAbsMax(d_vec+startIndex, SegmentLength);
}
#include "includes.h"
__device__ void MakeCountSegment_sharedAtomic(float *segment, int *bins, const int seglength, int *segCounter, int *s_counter, const int countlength, const float low, const float high, const float slope)
{
int bin;
float temp;
for (int jj=0; jj<seglength; jj++){
temp = abs(segment[jj]);
if ( ( temp > low ) & ( temp < high ) ) {
bin = (int)ceil(slope*abs(high-temp));
}
else if (temp >= high) {
bin = 0;
}
else bin = countlength - 1;
bins[jj]=bin;
atomicAdd(s_counter+bin,1);
}

for (int jj=0; jj<countlength; jj++) segCounter[jj]=s_counter[jj];

return;
}
__device__ void MakeCountSegment(float *segment, int *bins, const int seglength, int *segCounter, const int countlength, const float low, const float high, const float slope)
{
int bin;
float temp;
for (int jj=0; jj<seglength; jj++){
temp = abs(segment[jj]);
if ( ( temp > low ) & ( temp < high ) ) {
bin = (int)ceil(slope*abs(high-temp));
}
else if (temp >= high) {
bin = 0;
}
else bin = countlength - 1;
bins[jj]=bin;
segCounter[bin] = segCounter[bin] + 1;
}

return;
}
__global__ void make_and_count_seg_sharedAtomic(float *vec, int *bin, int *segcounter, const int length, const int countlength, const int HighLength, const int HighSegmentLength, const int threadsHigh, const int LowSegmentLength, const float low, const float high, const float slope)
{
int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
int startIndex, SegmentLength, startCountIndex;
extern __shared__ int s_counter[];

startCountIndex = xIndex*countlength;

if ( (xIndex*HighSegmentLength > HighLength) & ( (HighLength + (xIndex-threadsHigh+1)*LowSegmentLength) < length ) ){
startIndex = HighLength + (xIndex-threadsHigh)*LowSegmentLength;
SegmentLength = LowSegmentLength;
}
else {
startIndex = xIndex*HighSegmentLength;
SegmentLength = HighSegmentLength;
}
MakeCountSegment_sharedAtomic(vec+startIndex, bin+startIndex, SegmentLength, segcounter+startCountIndex, s_counter, countlength, low, high, slope);
}
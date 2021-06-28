#include "includes.h"


// Kind of lame, but just put static file-level variables here for now.
// Pointer to device results array.
float * dev_result = 0;

// Pointer to device data array.
float * dev_data = 0;

// Size of data/result sets (i.e. number of entries in array).
unsigned int testArraySize = 0;

// GPU function to converts the provided dBm value to mW.
// The power in milliwatts (P(mW)) is equal to 1mW times 10 raised by the
// power in decibel-milliwatts (P(dBm)) divided by 10:
// P(mW) = 1mW * 10 ^ (P(dBm) / 10)
__device__ float convertDbmToMw(const float dBm)
{
return powf(10.0f, dBm / 10.0f);
}
__global__ void convertDbmToMwKernal(float * result, const float * data)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
result[i] = convertDbmToMw(data[i]);
}
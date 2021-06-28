#include "includes.h"
__global__ void kernel( uchar4 *ptr, int ticks ) {
// map from threadIdx/BlockIdx to pixel position
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int offset = x + y * blockDim.x * gridDim.x;

// now calculate the value at that position
float fx = x - DIM/2;
float fy = y - DIM/2;
float d = sqrtf( fx * fx + fy * fy );
unsigned char grey = (unsigned char)(128.0f + 127.0f *
cos(d/10.0f - ticks/7.0f) /
(d/10.0f + 1.0f));
ptr[offset].x = grey;
ptr[offset].y = grey;
ptr[offset].z = grey;
ptr[offset].w = 255;
}
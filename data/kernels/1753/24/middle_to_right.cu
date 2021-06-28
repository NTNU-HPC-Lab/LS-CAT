#include "includes.h"
__global__ void middle_to_right(float* data, const int nx, const int ny)
{
float tmp;
for ( int r  = 0; r < ny; ++r ) {
float last_val = data[r*nx+nx/2];
for ( int c = nx-1; c >=  nx/2; --c ){
int idx = r*nx+c;
tmp = data[idx];
data[idx] = last_val;
last_val = tmp;
}
}
}
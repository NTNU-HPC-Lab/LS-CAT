#include "includes.h"
__global__ void middle_to_top(float* data, const int nx, const int ny)
{
float tmp;
for ( int c = 0; c < nx; ++c ) {
// Get the value in the top row
float last_val = data[ny/2*nx + c];
for ( int r = ny-1; r >= ny/2; --r ){
int idx = r*nx+c;
tmp = data[idx];
data[idx] = last_val;
last_val = tmp;
}
}
}
#include "includes.h"
__device__ double complexMagnitude(double2 in){
return sqrt(in.x*in.x + in.y*in.y);
}
__global__ void complexMag_test(double2 *in, double *out){
out[0] = complexMagnitude(in[0]);
}
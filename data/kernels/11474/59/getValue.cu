#include "includes.h"
__global__ void getValue(float4 *outdata, float *indata) {
// outdata[0] = indata[0];
float4 my4 = make_float4(indata[0], indata[3], indata[1], indata[2]);
outdata[0] = my4;
}
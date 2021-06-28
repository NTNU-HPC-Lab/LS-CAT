#include "includes.h"
__global__ void ReplaceNANs(float* buffer, float value, int size){
int offset = CUDASTDOFFSET;
float current = buffer[offset];
current = isfinite(current) ? current : value;
if(offset < size ) buffer[offset] = current;
}
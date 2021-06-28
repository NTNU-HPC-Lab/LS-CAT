#include "includes.h"
__global__ void getValueChar(char *outdata, char *indata) {
outdata[0] = indata[0] + 3;
}
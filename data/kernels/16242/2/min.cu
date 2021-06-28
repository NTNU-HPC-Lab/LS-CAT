#include "includes.h"
__global__ void min(int* U, int* d, int* outDel, int* minOutEdges, size_t gSize, int useD) {
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

int pos1 = 2*globalThreadId;
int pos2 = 2*globalThreadId + 1;
int val1, val2;
if(pos1 < gSize) {
val1 = minOutEdges[pos1] + (useD ? d[pos1] : 0);
if(pos2 < gSize) {
val2 = minOutEdges[pos2] + (useD ? d[pos2] : 0);

val1 = val1 <= 0 ? INT_MAX : val1;
val2 = val2 <= 0 ? INT_MAX : val2;
if(useD) {
val1 = U[pos1] ? val1 : INT_MAX;
val2 = U[pos2] ? val2 : INT_MAX;
}
if(val1 > val2) {
outDel[globalThreadId] = val2;
}
else{
outDel[globalThreadId] = val1;
}
}
else {
val1 = val1 <= 0 ? INT_MAX : val1;
if(useD) {
val1 = U[pos1] ? val1 : INT_MAX;
}
outDel[globalThreadId] = val1;
}
}
}
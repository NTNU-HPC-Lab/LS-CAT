#include "includes.h"

void printUsage(char* appName);
int parseArgs(char** argv,int* pSize,int *print,int argc);
uint64_t getTime();




__global__ void prime(int *a, int count)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;	//Handle the data at the index
if(tid > count) return;

int can = a[tid];
int counter=3;
//int flag=0;
//float limit = sqrtf((float)can);
float limit = sqrtf(can);
limit = limit+1;
// if even -- get out
if(can%2==0)
{
a[tid] = 1;
}
else
{
for(;counter<=limit;counter+=2)
{
if(can%counter==0)
{
a[tid]=1; // set as prime
break;
}
}
}
}
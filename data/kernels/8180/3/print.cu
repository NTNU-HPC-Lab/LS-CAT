#include "includes.h"
// richu shaji abraham richursa
using namespace std;
__global__ void print(int *d_predicateArrry,int numberOfElements)
{

for(int i=0;i<numberOfElements;i++)
{
printf("index = %d value = %d\n",i,d_predicateArrry[i]);
}
}
#include "includes.h"
__global__ void serialReduction(int *d_array, int numberOfElements)
{
int sum = 0;
for(int i=0;i<numberOfElements;i++)
{
sum = sum + d_array[i];
}
printf("%d",sum);
}
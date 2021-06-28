#include "includes.h"
__global__ void quickSort(int *x, int *dfirst, int *dlast, int *list)
{
int idx = threadIdx.x;
int first = dfirst[idx];
int last = dlast[idx];
list[idx] = 0;

if(first<last)
{
int pivot, j, temp, i;

pivot = first;
i = first;
j = last;

while(i<j)
{
while(x[i]<=x[pivot] && i<last)
i++;
while(x[j] > x[pivot])
j--;
if(i<j)
{
temp = x[i];
x[i] = x[j];
x[j] = temp;
}
}

temp = x[pivot];
x[pivot] = x[j];
x[j] = temp;

for(i=first; i<=last; i++)
if(x[i] > x[i+1])
{
list[idx] = j+1;
break;
}
}
}
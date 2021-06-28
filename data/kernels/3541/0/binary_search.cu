#include "includes.h"


__global__ void binary_search(int* a, int* b, bool* c, int sizeofa) //kernal function
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
printf(" %d\n", index);
int key = b[index];
int min = 0, max = sizeofa;
int mid = sizeofa / 2;
while (min != mid)
{
if (key == a[mid])
{
break;
}
else if (key < a[mid])
{
min = min;
max = mid;
}
else {
min = mid;
max = max;
}
mid = (min + max) / 2;
}

if (key == a[mid])
c[index] = true;
else
c[index] = false;

printf(" %d %d  %d %d\n", index, key, a[mid],c[index]);
}
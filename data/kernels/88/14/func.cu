#include "includes.h"
__global__ void func (char* stringInput, int stringSize, int* integerInput, char* dummySpace) {
int counter = 0;
for (int i=0;i<stringSize;i++)
dummySpace[counter++] = stringInput[i];

for (int i=0;i<sizeof(int);i++)
dummySpace[counter++] = ((char*)integerInput)[i];
}
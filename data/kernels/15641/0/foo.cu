#include "includes.h"
__global__ void foo() {
for(int i=0;i<1000;i++)
pow(2,32);
}
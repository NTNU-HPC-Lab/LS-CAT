#include "includes.h"
__global__ void use_ptr3() {
for (int i = 0; i < 100; i++)
const_ptr[i] = i;
}
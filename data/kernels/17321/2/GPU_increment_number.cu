#include "includes.h"
__global__ void GPU_increment_number(int* buffer, int initial)
{
buffer[0] = 1 + initial;
}
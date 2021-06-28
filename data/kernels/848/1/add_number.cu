#include "includes.h"
__global__ void add_number(float *ad,float *bd)
{

*ad += *bd;                             //adding values in GPU memory
}
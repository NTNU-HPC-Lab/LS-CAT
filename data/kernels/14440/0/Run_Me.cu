#include "includes.h"








__global__ void Run_Me( int* The_Array , int size)
{
int ID = blockIdx.x;
if(ID < 4)
The_Array[ID] = The_Array[ID] * The_Array[ID];

}
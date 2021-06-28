#include "includes.h"
__global__ void externSet(int* variablesMem,int* lastValuesMem, int nQueen,int nVariableCollection){

int index = threadIdx.x + blockIdx.x * blockDim.x;
if(index < nVariableCollection*nQueen*nQueen){
variablesMem[index] = 1;
if(index < nVariableCollection*nQueen)
lastValuesMem[index] = 0;
}

}
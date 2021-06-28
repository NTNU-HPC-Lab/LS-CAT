#include "includes.h"

using namespace std;

#define delta		     10
#define rows			 50
#define columns			 50


__global__ void SomeKernel(int* res, int* data, int col, int row,int y, int step)
{
unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
//Считаем идентификатор текущего потока
int currDelta = 0;
for (int i=step*threadId; (i<(threadId+1)*step) && (i < col); i++) //Работа со столбцами по потокам
{
for (int j = y; j > 0; j--) //Здесь работа со строками
{
currDelta = data[i + j*row] - data[i + (j-1)*row];
//если текущая разность больше дельты, то запоминаем у-координату
if( ( currDelta >= 0 ? currDelta : currDelta*-1 ) > 10){
res[i] = j-1;
break;
}
}
}
}
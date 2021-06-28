#include "includes.h"
__global__ void squareMatrixMulKernel(int *c, int *a, int *b, int arrayWidth)
{
float sum = 0;

//행렬에서 계산하려고 하는 위치의 인덱스 이것은 공식화 된것이므로 외우진 말자.
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;


//블록당 쓰레드가 4x4이고
//블록의 개수가 1x1이면
//printf("%d, %d / %d, %d / %d, %d\n", blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
// 4, 4, 0, 0, x, y 이렇게 앞에 4개의 숫자는 고정된 것을 볼 수 있었다.
//blockDim : 블록 안쪽에 포함된 쓰레드가 어떤 ㅁxㅁ 차원으로 되어있는지.
//blockIdx : 블록의 인덱스
//threadIdx : 쓰레드의 인덱스

for (int i = 0; i < arrayWidth; ++i)
{
float Aelement = a[row * arrayWidth + i];
float Belement = b[i*arrayWidth + col];
sum += Aelement * Belement;
}
c[row * arrayWidth + col] = sum;
}
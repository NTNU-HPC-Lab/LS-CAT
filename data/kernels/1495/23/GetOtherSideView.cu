#include "includes.h"
__global__ static void GetOtherSideView(float* Data, float* OtherSideData, int SizeX, int SizeY, int FinalSizeZ)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id >= SizeX * SizeY)
{
printf("範圍有錯!!\n");
return;
}

// id 換算
int idX = id / SizeY;
int idY = id % SizeY;
int DataOffsetIndex = idX * SizeY * FinalSizeZ + idY * FinalSizeZ;

// 總和一個 SizeZ
float totalZ = 0;
for (int i = 0; i < FinalSizeZ; i++)
totalZ += Data[DataOffsetIndex + i];


// 這邊的單位要調整一下
// rows => 是張樹 (SizeY)
// cols => 是 SizeX
int offsetIndex = idY * SizeX + idX;
OtherSideData[offsetIndex] = totalZ;
}
#include "includes.h"
__global__ static void ShiftFinalData(float* AfterFFTData, float* ShiftData, int SizeX, int SizeY, int FinalSizeZ, int FinalDataSize)
{
// 這邊要做位移
// 由於硬體是這樣子 ↓
// => | ->
// ("->" 是指第一段，"=>" 是指第二段)
int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
threadIdx.x;

if (id >= FinalDataSize)
{
printf("Shift Data 有錯誤!!\n");
return;
}

// 這邊的算法要對應回去原本的資料
int idZ = id % FinalSizeZ;
int tempIndex = id / FinalSizeZ;
int idX = tempIndex % SizeX;
int idY = tempIndex / SizeX;

// SizeY 折回來
// (0 ~ 124 125 ~ 249)
//		↓
// (125 ~ 249 0 ~ 124)
idY = (idY + SizeY / 2) % SizeY;

int NewIndex = idY * SizeX * FinalSizeZ + idX * FinalSizeZ + idZ;
ShiftData[id] = AfterFFTData[NewIndex];
//ShiftData[id] = AfterFFTData[id];
}
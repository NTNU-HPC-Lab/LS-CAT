#include "includes.h"
__global__ static void FrequencyAdjust(int* OCTData, float* KSpaceData, float* PXScale, int* IndexArray, int CutIndex, int SizeX, int SizeY, int SizeZ)
{
// 這邊是 Denoise，把兩個 Channel 的資料相加
int id = blockIdx.y * gridDim.x * gridDim.z * blockDim.x +			// Y	=> Y * 250 * (2 * 1024)
blockIdx.x * gridDim.z * blockDim.x +							// X	=> X * (2 * 1024)
blockIdx.z * blockDim.x +										// Z	=> (Z1 * 1024 + Z2)
threadIdx.x;

if (id >= SizeX * SizeY * SizeZ)
{
printf("Frequency 轉換的地方有問題");
return;
}

// 算回原本的 Index
int idZ = id % SizeZ;
if (IndexArray[idZ] == -1 || idZ >= CutIndex || idZ == 0)
{
KSpaceData[id] = 0;
return;
}

// 要算斜率前，先拿出上一筆資料
int LastPXScaleIndex = (IndexArray[idZ] - 1 <= 0 ? 0 : IndexArray[idZ] - 1);

double m = (double)(OCTData[id] - OCTData[id - 1]) / (PXScale[IndexArray[idZ]] - PXScale[LastPXScaleIndex]);
double c = OCTData[id] - m * PXScale[IndexArray[idZ]];
KSpaceData[id] = m * idZ + c;
}
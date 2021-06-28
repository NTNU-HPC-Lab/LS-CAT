#include "includes.h"
__device__ float3 operator % (const float3 & v1, const float3 & v2)
{   return make_float3(v1.y*v2.z - v1.z*v2.y,
v1.z*v2.x - v1.x*v2.z,
v1.x*v2.y - v1.y*v2.x);}
__global__ void CopyOutBack(float4* d_tetT, float* d_vertT, int* d_vertMem, int* d_vertMemOutside, int* d_BlockSizes, int* d_ActiveList,int m_maxNumInVert,int m_maxNumTotalTets, int m_maxVertMappingInside, int m_maxVertMappingOutside)
{
int list_idx = blockIdx.y*gridDim.x + blockIdx.x;
// retrieve actual block index from the active list
int block_idx = d_ActiveList[list_idx];
int block_size = d_BlockSizes[block_idx];

////////////initialize shared memory//////////////////////////////////////////

int tx = threadIdx.x;
int tet_base = block_idx*m_maxNumTotalTets;
int vert_base = block_idx*m_maxNumInVert;
int tmpindex;

if(tx < block_size)
{
float T = d_vertT[vert_base + tx];

int j =0;
tmpindex = d_vertMem[block_idx*m_maxVertMappingInside*m_maxNumInVert +  tx * m_maxVertMappingInside + j];
while(j < m_maxVertMappingInside && (tmpindex - tet_base*4) > -1) // update gloal memory inside all the old to the min
{
int segment = tmpindex / 4;
int offset = tmpindex % 4;
switch(offset)
{
case 0:
d_tetT[segment].x = T;
break;
case 1:
d_tetT[segment].y = T;
break;
case 2:
d_tetT[segment].z = T;
break;
case 3:
d_tetT[segment].w = T;
break;
}
j++;
tmpindex = d_vertMem[block_idx*m_maxVertMappingInside*m_maxNumInVert +  tx * m_maxVertMappingInside + j];
}

for(int j = 0; (j < m_maxVertMappingOutside) && ((tmpindex = d_vertMemOutside[block_idx*m_maxVertMappingOutside*m_maxNumInVert +  tx * m_maxVertMappingOutside + j]) > -1 ); j++) // update gloal memory outside all the old to the min
{
int segment = tmpindex / 4;
int offset = tmpindex % 4;
switch(offset)
{
case 0:
d_tetT[segment].x = T;
break;
case 1:
d_tetT[segment].y = T;
break;
case 2:
d_tetT[segment].z = T;
break;
case 3:
d_tetT[segment].w = T;
break;
}
}
}
}
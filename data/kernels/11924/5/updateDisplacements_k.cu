#include "includes.h"
__global__ void updateDisplacements_k(float4 *Ui_t, float4 *Ui_tminusdt, float *M, float4 *Ri, float4 *Fi, int maxNumForces, float4 *ABC, unsigned int numPoints)
{
int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

if (me_idx>=numPoints)
return;

float4 F = make_float4(0,0,0,0);

//	printf("Max num forces: %i\n", maxNumForces);

for (int i=0; i<maxNumForces; i++)
{
float4 force_to_add = Fi[me_idx*maxNumForces+i];
F.x += force_to_add.x;
F.y += force_to_add.y;
F.z += force_to_add.z;
}
//	printf("Accumulated node %i force: %f, %f, %f \n", me_idx, F.x, F.y, F.z);

float4 ABCi = ABC[me_idx];
float4 Uit = Ui_t[me_idx];
float4 Uitminusdt = Ui_tminusdt[me_idx];

float4 R = Ri[me_idx];
float x = ABCi.x * (R.x - F.x) + ABCi.y * Uit.x + ABCi.z * Uitminusdt.x;
float y = ABCi.x * (R.y - F.y) + ABCi.y * Uit.y + ABCi.z * Uitminusdt.y;
float z = ABCi.x * (R.z - F.z) + ABCi.y * Uit.z + ABCi.z * Uitminusdt.z;

/*	float x = ABCi.x * (-F.x) + ABCi.y * Ui_t[me_idx].x + ABCi.z * Ui_tminusdt[me_idx].x;
float y = ABCi.x * (-F.x) + ABCi.y * Ui_t[me_idx].y + ABCi.z * Ui_tminusdt[me_idx].y;
float z = ABCi.x * (-F.x ) + ABCi.y * Ui_t[me_idx].z + ABCi.z * Ui_tminusdt[me_idx].z;
*/
Ui_tminusdt[me_idx] = make_float4(x,y,z,0);//XXXXXXXXXXXXXXXXXXXXX

}
#include "includes.h"
////////////////////////////////////////////////////////////////////////////////
/*
Hologram generating algorithms for CUDA Devices

Copyright 2009, 2010, 2011, 2012 Martin Persson
martin.persson@physics.gu.se

This file is part of GenerateHologramCUDA.

GenerateHologramCUDA is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GenerateHologramCUDA is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with GenerateHologramCUDA.  If not, see <http://www.gnu.org/licenses/>.
*/
///////////////////////////////////////////////////////////////////////////////////
//The function "GenerateHologram" contains two different algorithms for
//hologram generation. The last parameter in the function call selects which
//one to use:
//0: Complex addition of "Lenses and Prisms", no optimization (3D)
//1: Weighted Gerchberg-Saxton algorithm using Fresnel propagation (3D)
//2: Weighted Gerchberg-Saxton algorithm using Fast Fourier Transforms (2D)
//-(0) produces optimal holograms for 1 or 2 traps and is significantly faster.
//     (0) is automatically selected if the number of spots is < 3.
////////////////////////////////////////////////////////////////////////////////
//Fresnel propagation based algorithm (1) described in:
//Roberto Di Leonardo, Francesca Ianni, and Giancarlo Ruocco
//"Computer generation of optimal holograms for optical trap arrays"
//Opt. Express 15, 1913-1922 (2007)
//
//The original algorithm has been modified to allow variable spot amplitudes
////////////////////////////////////////////////////////////////////////////////
//Naming convention for variables:
//-The prefix indicates where data is located
//--In host functions:		h = host memory
//				d = device memory
//				c = constant memory
//--In global functions:	g = global memory
//				s = shared memory
//				c = constant memory
//				no prefix = registers
//-The suffix indicates the data type, no suffix usually indicates an iteger
////////////////////////////////////////////////////////////////////////////////
//Possible improvements:
//-Improve convergence of the GS algorithms for 2 spots.							*done
//-Compensate spot intensities for distance from center of field.					*done
//-Put all arguments for device functions and trap positions in constant memory.	*done
// (Requires all functions to be moved into the same file or the use of some
// workaround found on nVidia forum)
//-Put pSLMstart and aLaser in texture memory (may not improve performance on Fermi devices)
//-Use "zero-copy" to transfer pSLM to host.
//-Rename functions and variables for consistency and readability
//-Allow variable spot phases for Lenses and Prisms
////////////////////////////////////////////////////////////////////////////////

//#define M_CUDA_DEBUG			   //activates a number of custom debug macros//
float dt_milliseconds;
cudaEvent_t start, stop;
////////////////////////////////////////////////////////////////////////////////
//Includes
////////////////////////////////////////////////////////////////////////////////

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define MAX_SPOTS 1024	//decrease this if your GPU keeps running out of memory
#define BLOCK_SIZE 256	//should be a power of 2
#define SLM_SIZE 512
#if ((SLM_SIZE==16)||(SLM_SIZE==32)||(SLM_SIZE==64)||(SLM_SIZE==128)||(SLM_SIZE==256)||(SLM_SIZE==512)||(SLM_SIZE==1024)||(SLM_SIZE==2048))
#define SLMPOW2			//Uses bitwise modulu operations if the SLM size is a power of 2
#endif

////////////////////////////////////////////////////////////////////////////////
// forward declarations
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//Set correction parameters
////////////////////////////////////////////////////////////////////////////////
__device__ unsigned char applyPolLUT(float phase2pi, float X, float Y, float *s_c)
{
switch (c_N_PolLUTCoeff[0])	{
case 120:
return (unsigned char)(s_c[0] + s_c[1]*X + s_c[2]*Y + s_c[3]*phase2pi + s_c[4]*X*X + s_c[5]*X*Y + s_c[6]*X*phase2pi + s_c[7]*Y*Y + s_c[8]*Y*phase2pi + s_c[9]*phase2pi*phase2pi + s_c[10]*X*X*X + s_c[11]*X*X*Y + s_c[12]*X*X*phase2pi + s_c[13]*X*Y*Y + s_c[14]*X*Y*phase2pi + s_c[15]*X*phase2pi*phase2pi + s_c[16]*Y*Y*Y + s_c[17]*Y*Y*phase2pi + s_c[18]*Y*phase2pi*phase2pi + s_c[19]*phase2pi*phase2pi*phase2pi + s_c[20]*X*X*X*X + s_c[21]*X*X*X*Y + s_c[22]*X*X*X*phase2pi + s_c[23]*X*X*Y*Y + s_c[24]*X*X*Y*phase2pi + s_c[25]*X*X*phase2pi*phase2pi + s_c[26]*X*Y*Y*Y + s_c[27]*X*Y*Y*phase2pi + s_c[28]*X*Y*phase2pi*phase2pi + s_c[29]*X*phase2pi*phase2pi*phase2pi + s_c[30]*Y*Y*Y*Y + s_c[31]*Y*Y*Y*phase2pi + s_c[32]*Y*Y*phase2pi*phase2pi + s_c[33]*Y*phase2pi*phase2pi*phase2pi + s_c[34]*phase2pi*phase2pi*phase2pi*phase2pi + s_c[35]*X*X*X*X*X + s_c[36]*X*X*X*X*Y + s_c[37]*X*X*X*X*phase2pi + s_c[38]*X*X*X*Y*Y + s_c[39]*X*X*X*Y*phase2pi + s_c[40]*X*X*X*phase2pi*phase2pi + s_c[41]*X*X*Y*Y*Y + s_c[42]*X*X*Y*Y*phase2pi + s_c[43]*X*X*Y*phase2pi*phase2pi + s_c[44]*X*X*phase2pi*phase2pi*phase2pi + s_c[45]*X*Y*Y*Y*Y + s_c[46]*X*Y*Y*Y*phase2pi + s_c[47]*X*Y*Y*phase2pi*phase2pi + s_c[48]*X*Y*phase2pi*phase2pi*phase2pi + s_c[49]*X*phase2pi*phase2pi*phase2pi*phase2pi + s_c[50]*Y*Y*Y*Y*Y + s_c[51]*Y*Y*Y*Y*phase2pi + s_c[52]*Y*Y*Y*phase2pi*phase2pi + s_c[53]*Y*Y*phase2pi*phase2pi*phase2pi + s_c[54]*Y*phase2pi*phase2pi*phase2pi*phase2pi + s_c[55]*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[56]*X*X*X*X*X*X + s_c[57]*X*X*X*X*X*Y + s_c[58]*X*X*X*X*X*phase2pi + s_c[59]*X*X*X*X*Y*Y + s_c[60]*X*X*X*X*Y*phase2pi + s_c[61]*X*X*X*X*phase2pi*phase2pi + s_c[62]*X*X*X*Y*Y*Y + s_c[63]*X*X*X*Y*Y*phase2pi + s_c[64]*X*X*X*Y*phase2pi*phase2pi + s_c[65]*X*X*X*phase2pi*phase2pi*phase2pi + s_c[66]*X*X*Y*Y*Y*Y + s_c[67]*X*X*Y*Y*Y*phase2pi + s_c[68]*X*X*Y*Y*phase2pi*phase2pi + s_c[69]*X*X*Y*phase2pi*phase2pi*phase2pi + s_c[70]*X*X*phase2pi*phase2pi*phase2pi*phase2pi + s_c[71]*X*Y*Y*Y*Y*Y + s_c[72]*X*Y*Y*Y*Y*phase2pi + s_c[73]*X*Y*Y*Y*phase2pi*phase2pi + s_c[74]*X*Y*Y*phase2pi*phase2pi*phase2pi + s_c[75]*X*Y*phase2pi*phase2pi*phase2pi*phase2pi + s_c[76]*X*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[77]*Y*Y*Y*Y*Y*Y + s_c[78]*Y*Y*Y*Y*Y*phase2pi + s_c[79]*Y*Y*Y*Y*phase2pi*phase2pi + s_c[80]*Y*Y*Y*phase2pi*phase2pi*phase2pi + s_c[81]*Y*Y*phase2pi*phase2pi*phase2pi*phase2pi + s_c[82]*Y*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[83]*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[84]*X*X*X*X*X*X*X + s_c[85]*X*X*X*X*X*X*Y + s_c[86]*X*X*X*X*X*X*phase2pi + s_c[87]*X*X*X*X*X*Y*Y + s_c[88]*X*X*X*X*X*Y*phase2pi + s_c[89]*X*X*X*X*X*phase2pi*phase2pi + s_c[90]*X*X*X*X*Y*Y*Y + s_c[91]*X*X*X*X*Y*Y*phase2pi + s_c[92]*X*X*X*X*Y*phase2pi*phase2pi + s_c[93]*X*X*X*X*phase2pi*phase2pi*phase2pi + s_c[94]*X*X*X*Y*Y*Y*Y + s_c[95]*X*X*X*Y*Y*Y*phase2pi + s_c[96]*X*X*X*Y*Y*phase2pi*phase2pi + s_c[97]*X*X*X*Y*phase2pi*phase2pi*phase2pi + s_c[98]*X*X*X*phase2pi*phase2pi*phase2pi*phase2pi + s_c[99]*X*X*Y*Y*Y*Y*Y + s_c[100]*X*X*Y*Y*Y*Y*phase2pi + s_c[101]*X*X*Y*Y*Y*phase2pi*phase2pi + s_c[102]*X*X*Y*Y*phase2pi*phase2pi*phase2pi + s_c[103]*X*X*Y*phase2pi*phase2pi*phase2pi*phase2pi + s_c[104]*X*X*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[105]*X*Y*Y*Y*Y*Y*Y + s_c[106]*X*Y*Y*Y*Y*Y*phase2pi + s_c[107]*X*Y*Y*Y*Y*phase2pi*phase2pi + s_c[108]*X*Y*Y*Y*phase2pi*phase2pi*phase2pi + s_c[109]*X*Y*Y*phase2pi*phase2pi*phase2pi*phase2pi + s_c[110]*X*Y*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[111]*X*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[112]*Y*Y*Y*Y*Y*Y*Y + s_c[113]*Y*Y*Y*Y*Y*Y*phase2pi + s_c[114]*Y*Y*Y*Y*Y*phase2pi*phase2pi + s_c[115]*Y*Y*Y*Y*phase2pi*phase2pi*phase2pi + s_c[116]*Y*Y*Y*phase2pi*phase2pi*phase2pi*phase2pi + s_c[117]*Y*Y*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[118]*Y*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[119]*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi);
case 84:
return (unsigned char)(s_c[0] + s_c[1]*X + s_c[2]*Y + s_c[3]*phase2pi + s_c[4]*X*X + s_c[5]*X*Y + s_c[6]*X*phase2pi + s_c[7]*Y*Y + s_c[8]*Y*phase2pi + s_c[9]*phase2pi*phase2pi + s_c[10]*X*X*X + s_c[11]*X*X*Y + s_c[12]*X*X*phase2pi + s_c[13]*X*Y*Y + s_c[14]*X*Y*phase2pi + s_c[15]*X*phase2pi*phase2pi + s_c[16]*Y*Y*Y + s_c[17]*Y*Y*phase2pi + s_c[18]*Y*phase2pi*phase2pi + s_c[19]*phase2pi*phase2pi*phase2pi + s_c[20]*X*X*X*X + s_c[21]*X*X*X*Y + s_c[22]*X*X*X*phase2pi + s_c[23]*X*X*Y*Y + s_c[24]*X*X*Y*phase2pi + s_c[25]*X*X*phase2pi*phase2pi + s_c[26]*X*Y*Y*Y + s_c[27]*X*Y*Y*phase2pi + s_c[28]*X*Y*phase2pi*phase2pi + s_c[29]*X*phase2pi*phase2pi*phase2pi + s_c[30]*Y*Y*Y*Y + s_c[31]*Y*Y*Y*phase2pi + s_c[32]*Y*Y*phase2pi*phase2pi + s_c[33]*Y*phase2pi*phase2pi*phase2pi + s_c[34]*phase2pi*phase2pi*phase2pi*phase2pi + s_c[35]*X*X*X*X*X + s_c[36]*X*X*X*X*Y + s_c[37]*X*X*X*X*phase2pi + s_c[38]*X*X*X*Y*Y + s_c[39]*X*X*X*Y*phase2pi + s_c[40]*X*X*X*phase2pi*phase2pi + s_c[41]*X*X*Y*Y*Y + s_c[42]*X*X*Y*Y*phase2pi + s_c[43]*X*X*Y*phase2pi*phase2pi + s_c[44]*X*X*phase2pi*phase2pi*phase2pi + s_c[45]*X*Y*Y*Y*Y + s_c[46]*X*Y*Y*Y*phase2pi + s_c[47]*X*Y*Y*phase2pi*phase2pi + s_c[48]*X*Y*phase2pi*phase2pi*phase2pi + s_c[49]*X*phase2pi*phase2pi*phase2pi*phase2pi + s_c[50]*Y*Y*Y*Y*Y + s_c[51]*Y*Y*Y*Y*phase2pi + s_c[52]*Y*Y*Y*phase2pi*phase2pi + s_c[53]*Y*Y*phase2pi*phase2pi*phase2pi + s_c[54]*Y*phase2pi*phase2pi*phase2pi*phase2pi + s_c[55]*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[56]*X*X*X*X*X*X + s_c[57]*X*X*X*X*X*Y + s_c[58]*X*X*X*X*X*phase2pi + s_c[59]*X*X*X*X*Y*Y + s_c[60]*X*X*X*X*Y*phase2pi + s_c[61]*X*X*X*X*phase2pi*phase2pi + s_c[62]*X*X*X*Y*Y*Y + s_c[63]*X*X*X*Y*Y*phase2pi + s_c[64]*X*X*X*Y*phase2pi*phase2pi + s_c[65]*X*X*X*phase2pi*phase2pi*phase2pi + s_c[66]*X*X*Y*Y*Y*Y + s_c[67]*X*X*Y*Y*Y*phase2pi + s_c[68]*X*X*Y*Y*phase2pi*phase2pi + s_c[69]*X*X*Y*phase2pi*phase2pi*phase2pi + s_c[70]*X*X*phase2pi*phase2pi*phase2pi*phase2pi + s_c[71]*X*Y*Y*Y*Y*Y + s_c[72]*X*Y*Y*Y*Y*phase2pi + s_c[73]*X*Y*Y*Y*phase2pi*phase2pi + s_c[74]*X*Y*Y*phase2pi*phase2pi*phase2pi + s_c[75]*X*Y*phase2pi*phase2pi*phase2pi*phase2pi + s_c[76]*X*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[77]*Y*Y*Y*Y*Y*Y + s_c[78]*Y*Y*Y*Y*Y*phase2pi + s_c[79]*Y*Y*Y*Y*phase2pi*phase2pi + s_c[80]*Y*Y*Y*phase2pi*phase2pi*phase2pi + s_c[81]*Y*Y*phase2pi*phase2pi*phase2pi*phase2pi + s_c[82]*Y*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi + s_c[83]*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi);
case 56:
return (unsigned char)(s_c[0] + s_c[1]*X + s_c[2]*Y + s_c[3]*phase2pi + s_c[4]*X*X + s_c[5]*X*Y + s_c[6]*X*phase2pi + s_c[7]*Y*Y + s_c[8]*Y*phase2pi + s_c[9]*phase2pi*phase2pi + s_c[10]*X*X*X + s_c[11]*X*X*Y + s_c[12]*X*X*phase2pi + s_c[13]*X*Y*Y + s_c[14]*X*Y*phase2pi + s_c[15]*X*phase2pi*phase2pi + s_c[16]*Y*Y*Y + s_c[17]*Y*Y*phase2pi + s_c[18]*Y*phase2pi*phase2pi + s_c[19]*phase2pi*phase2pi*phase2pi + s_c[20]*X*X*X*X + s_c[21]*X*X*X*Y + s_c[22]*X*X*X*phase2pi + s_c[23]*X*X*Y*Y + s_c[24]*X*X*Y*phase2pi + s_c[25]*X*X*phase2pi*phase2pi + s_c[26]*X*Y*Y*Y + s_c[27]*X*Y*Y*phase2pi + s_c[28]*X*Y*phase2pi*phase2pi + s_c[29]*X*phase2pi*phase2pi*phase2pi + s_c[30]*Y*Y*Y*Y + s_c[31]*Y*Y*Y*phase2pi + s_c[32]*Y*Y*phase2pi*phase2pi + s_c[33]*Y*phase2pi*phase2pi*phase2pi + s_c[34]*phase2pi*phase2pi*phase2pi*phase2pi + s_c[35]*X*X*X*X*X + s_c[36]*X*X*X*X*Y + s_c[37]*X*X*X*X*phase2pi + s_c[38]*X*X*X*Y*Y + s_c[39]*X*X*X*Y*phase2pi + s_c[40]*X*X*X*phase2pi*phase2pi + s_c[41]*X*X*Y*Y*Y + s_c[42]*X*X*Y*Y*phase2pi + s_c[43]*X*X*Y*phase2pi*phase2pi + s_c[44]*X*X*phase2pi*phase2pi*phase2pi + s_c[45]*X*Y*Y*Y*Y + s_c[46]*X*Y*Y*Y*phase2pi + s_c[47]*X*Y*Y*phase2pi*phase2pi + s_c[48]*X*Y*phase2pi*phase2pi*phase2pi + s_c[49]*X*phase2pi*phase2pi*phase2pi*phase2pi + s_c[50]*Y*Y*Y*Y*Y + s_c[51]*Y*Y*Y*Y*phase2pi + s_c[52]*Y*Y*Y*phase2pi*phase2pi + s_c[53]*Y*Y*phase2pi*phase2pi*phase2pi + s_c[54]*Y*phase2pi*phase2pi*phase2pi*phase2pi + s_c[55]*phase2pi*phase2pi*phase2pi*phase2pi*phase2pi);
case 35:
return (unsigned char)(s_c[0] + s_c[1]*X + s_c[2]*Y + s_c[3]*phase2pi + s_c[4]*X*X + s_c[5]*X*Y + s_c[6]*X*phase2pi + s_c[7]*Y*Y + s_c[8]*Y*phase2pi + s_c[9]*phase2pi*phase2pi + s_c[10]*X*X*X + s_c[11]*X*X*Y + s_c[12]*X*X*phase2pi + s_c[13]*X*Y*Y + s_c[14]*X*Y*phase2pi + s_c[15]*X*phase2pi*phase2pi + s_c[16]*Y*Y*Y + s_c[17]*Y*Y*phase2pi + s_c[18]*Y*phase2pi*phase2pi + s_c[19]*phase2pi*phase2pi*phase2pi + s_c[20]*X*X*X*X + s_c[21]*X*X*X*Y + s_c[22]*X*X*X*phase2pi + s_c[23]*X*X*Y*Y + s_c[24]*X*X*Y*phase2pi + s_c[25]*X*X*phase2pi*phase2pi + s_c[26]*X*Y*Y*Y + s_c[27]*X*Y*Y*phase2pi + s_c[28]*X*Y*phase2pi*phase2pi + s_c[29]*X*phase2pi*phase2pi*phase2pi + s_c[30]*Y*Y*Y*Y + s_c[31]*Y*Y*Y*phase2pi + s_c[32]*Y*Y*phase2pi*phase2pi + s_c[33]*Y*phase2pi*phase2pi*phase2pi + s_c[34]*phase2pi*phase2pi*phase2pi*phase2pi);
case 20:
return (unsigned char)(s_c[0] + s_c[1]*X + s_c[2]*Y + s_c[3]*phase2pi + s_c[4]*X*X + s_c[5]*X*Y + s_c[6]*X*phase2pi + s_c[7]*Y*Y + s_c[8]*Y*phase2pi + s_c[9]*phase2pi*phase2pi + s_c[10]*X*X*X + s_c[11]*X*X*Y + s_c[12]*X*X*phase2pi + s_c[13]*X*Y*Y + s_c[14]*X*Y*phase2pi + s_c[15]*X*phase2pi*phase2pi + s_c[16]*Y*Y*Y + s_c[17]*Y*Y*phase2pi + s_c[18]*Y*phase2pi*phase2pi + s_c[19]*phase2pi*phase2pi*phase2pi);
default:
return 0;
}
}
__device__ int getYint(int index, int X_int)
{
#ifdef SLMPOW2
int Y_int = (index-X_int)>>c_log2data_w[0];
#else
int Y_int = (float)(floor((float)index/c_data_w_f[0]));
#endif
return Y_int;
}
__device__ int getXint(int index)
{
#ifdef SLMPOW2
int X_int = index&(c_data_w[0]-1);
#else
float X_int= index%c_data_w[0];
#endif
return X_int;
}
__device__ float ApplyAberrationCorrection(float pSpot, float correction)
{
pSpot = pSpot - correction;		//apply correction
return (pSpot - (2.0f*M_PI) * floor((pSpot+M_PI) / (2.0f*M_PI))); //apply mod([-pi, pi], pSpot)
}
__device__ int phase2int32(float phase2pi)
{
return (int)floor((phase2pi + M_PI)*256.0f / (2.0f * M_PI));
}
__device__ unsigned char phase2uc(float phase2pi)
{
return (unsigned char)floor((phase2pi + M_PI)*256.0f / (2.0f * M_PI));
}
__device__ float uc2phase(float uc)
{
return (float)uc*2.0f*M_PI/256.0f - M_PI;
}
__global__ void ApplyCorrections(unsigned char *g_pSLM_uc, unsigned char *g_LUT, float *g_AberrationCorr_f, float *g_LUTPolCoeff_f)
{
int tid = threadIdx.x;
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float pSLM2pi_f = uc2phase(g_pSLM_uc[idx]);
if (c_useAberrationCorr_b[0])
pSLM2pi_f = ApplyAberrationCorrection(pSLM2pi_f, g_AberrationCorr_f[idx]);

if (c_usePolLUT_b[0])
{
int X_int = getXint(idx);
int Y_int = getYint(idx, X_int);
float X = c_SLMpitch_f[0]*(X_int - c_half_w_f[0]);
float Y = c_SLMpitch_f[0]*(Y_int - c_half_w_f[0]);
__shared__ float s_LUTcoeff[120];
if (tid < c_N_PolLUTCoeff[0])
s_LUTcoeff[tid] = g_LUTPolCoeff_f[tid];
__syncthreads();
g_pSLM_uc[idx] = applyPolLUT(pSLM2pi_f, X, Y, s_LUTcoeff);
}
else if (c_applyLUT_b[0])
{
__shared__ unsigned char s_LUT[256];
if (tid < 256)
s_LUT[tid] = g_LUT[tid];
__syncthreads();
g_pSLM_uc[idx] = s_LUT[phase2int32(pSLM2pi_f)];
}
else
g_pSLM_uc[idx] = phase2uc(pSLM2pi_f);
}
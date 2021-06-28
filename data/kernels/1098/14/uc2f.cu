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
__global__ void uc2f(float *f, unsigned char *uc, int N)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx<N)
{
f[idx] = uc[idx]*2.0f*M_PI/256.0f - M_PI;
}
}
#include "includes.h"
/*
* G2S
* Copyright (C) 2018, Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL (University of Lausanne)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//To use only for debugging purpose


#define PARTIAL_FFT

#ifndef FFTW_PLAN_OPTION
//FFTW_PATIENT
#define FFTW_PLAN_OPTION FFTW_ESTIMATE
#endif
// #if __cilk
// 	#define fillVectorized(name, begin, amount, value) name[begin:amount]=value;
// #else
#define fillVectorized(name, begin, amount, value) std::fill(name+begin,name+begin+amount,value);
// #endif


__global__ void compensateMissingDatakernel(float* errosArray, float* crossErrosArray, const unsigned int size, float val){
int j = blockIdx.x*blockDim.x + threadIdx.x;
if (j < size)
{
errosArray[j]=-std::fabs(errosArray[j]/(crossErrosArray[j]*crossErrosArray[j]*crossErrosArray[j]*crossErrosArray[j]));
if(crossErrosArray[j]==0.0f) errosArray[j]=val;
}
}
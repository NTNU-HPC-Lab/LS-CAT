#include "includes.h"
/******************************************************************************
* © Mathias Bourgoin, Université Pierre et Marie Curie (2011)
*
* Mathias.Bourgoin@gmail.com
*
* This software is a computer program whose purpose is allow GPU programming
* with the OCaml language.
*
* This software is governed by the CeCILL-B license under French law and
* abiding by the rules of distribution of free software.  You can  use,
* modify and/ or redistribute the software under the terms of the CeCILL-B
* license as circulated by CEA, CNRS and INRIA at the following URL
* "http://www.cecill.info".
*
* As a counterpart to the access to the source code and  rights to copy,
* modify and redistribute granted by the license, users are provided only
* with a limited warranty  and the software's author,  the holder of the
* economic rights,  and the successive licensors  have only  limited
* liability.
*
* In this respect, the user's attention is drawn to the risks associated
* with loading,  using,  modifying and/or developing or reproducing the
* software by the user in light of its specific status of free software,
* that may mean  that it is complicated to manipulate,  and  that  also
* therefore means  that it is reserved for developers  and  experienced
* professionals having in-depth computer knowledge. Users are therefore
* encouraged to load and test the software's suitability as regards their
* requirements in conditions enabling the security of their systems and/or
* data to be ensured and,  more generally, to use and operate it in the
* same conditions as regards security.
*
* The fact that you are presently reading this means that you have had
* knowledge of the CeCILL-B license and that you accept its terms.
*
* NOTE:  This file contains source code provided by NVIDIA Corporation.
*******************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

/****** Single precision *****/





/****** Double precision *****/







#ifdef __cplusplus
}
#endif
__global__ void int_bubble_filter( int* input, const int* vec1, int* output, const int count)
{
int i;
int k = 1;
int tid = blockDim.x * blockIdx.x + threadIdx.x;
if (tid <= count/2)
{
output[tid*2] = vec1[tid*2];
output[tid*2+1] = vec1[tid*2+1];
//barrier(CLK_GLOBAL_MEM_FENCE);

for (int n = 0; n < count*2; n++)
{
k = (k)?0:1;
i = (tid*2) + k;
if( i+1 < count)
{
if ((!input[i]) && (input[i+1]))
{
input[i] = 1;
input[i+1] = 0;
output[i] = output[i+1];
output[i+1] = 0;
}
else
{
if (!input[i])
output[i] = 0;
if (!input[i+1])
output[i+1] = 0;
}
}
__syncthreads();
}
}
}
#include "includes.h"
// ******************************************************************************************************
// PURPOSE		:	Print thread IDs for the 256 threads of a 2D configuration (16 * 16)				*
// LANGUAGE		:		CUDA C / CUDA C++																*
// ASSUMPTIONS	:	2D Configuration 16 threads in each x & y directions with thread block of (8*8)		*
//					threadIdx.z value will be zero since it is 2D configuration							*
// DATE			:	23 March 2020																		*
// AUTHOR		:	Vaibhav BENDRE 																		*
//					vaibhav.bendre7520@gmail.com														*
// ******************************************************************************************************




__global__ void printThreadIDs() {

printf("\n threadIdx.x : %d,   threadIdx.y :  %d ",threadIdx.x,threadIdx.y);

}
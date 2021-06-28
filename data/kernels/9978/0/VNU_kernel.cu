#include "includes.h"

#ifndef _VNU_KERNEL_H_
#define _VNU_KERNEL_H_


#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#define database_character( index)  CUT_BANK_CHECKER(database_character, index)
#define temp_1( index)              CUT_BANK_CHECKER(temp_1,             index)
#define temp_2( index)              CUT_BANK_CHECKER(temp_2,             index)



/*_________________________________________________Kernel_____________________________________________________*/



#endif // #ifndef _VNU_KERNEL_H_



/*_____________________________________________Begin CN Kernel___________________________________________________*/
#ifndef _CNU_KERNEL_H_
#define _CNU_KERNEL_H_


#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#define database_character( index)  CUT_BANK_CHECKER(database_character, index)
#define temp_1( index)              CUT_BANK_CHECKER(temp_1,             index)
#define temp_2( index)              CUT_BANK_CHECKER(temp_2,             index)

#endif // #ifndef _CNU_KERNEL_H_
__global__ void VNU_kernel(short int* device_array, short int* offset_array, short int* sign_array, short int* results_array)
{


/*_________________________________________Shared Memory Allocation____________________________________________*/

__shared__ short int  offset;                     // Memory offset values to be read from global memory
__shared__ short int thread_Id;
__shared__ short int current_Index;
/*_____________________________________Get access to thread ID and Block ID____________________________________*/

// access current thread id
thread_Id = threadIdx.x;

// Index for global memory
current_Index = ((blockIdx.x * blockDim.x + thread_Id)*2);


/*__________________________Each Thread gets its global memory variables and index(offset)_____________________*/

// Get offsets from global memory... currently these are set to zero for simplicity
offset = offset_array[current_Index];


/*_______________________________________________Begin VN_______________________________________________________*/



short int sign = 0;
short int input1 = results_array[current_Index + offset];
short int input2 = results_array[current_Index + offset +(1)];
short int input3 = device_array[(current_Index/2) + offset];

short int sum = (input1 + input2 + input3);

short int output1 = (sum - input1);
short int output2 = (sum - input2);

if(sum < 0){
sign = 1;
}

/*_________________________________Record Results back to Device Memory________________________________________*/
//Write outputs to the same addresses read initially from the global memory to get the input integers

results_array[current_Index + offset] = output1;
results_array[current_Index + offset +(1)] = output2;
sign_array[current_Index + offset] = sign;
sign_array[current_Index + offset +(1)] = sign;
}
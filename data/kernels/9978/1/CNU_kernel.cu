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
__global__ void CNU_kernel(short int* offset_array, short int* results_array)
{

/*_________________________________________Shared Memory Allocation____________________________________________*/

__shared__ short int  offset;                     // Memory offset values to be read from global memory
__shared__ short int thread_Id;
__shared__ short int current_Index;
/*_____________________________________Get access to thread ID and Block ID____________________________________*/

// access current thread id
thread_Id = threadIdx.x;

// Index for global memory
current_Index = ((blockIdx.x * blockDim.x + thread_Id)*3);


/*__________________________Each Thread gets its global memory variables and index(offset)_____________________*/

// Get offsets from global memory... currently these are set to zero for simplicity
offset = offset_array[current_Index];




/*___________________________________________CN Kernel Logic______________________________________________________*/

short int input1 = results_array[current_Index + offset];
short int input2 = results_array[current_Index + offset +(1)];
short int input3 = results_array[current_Index + offset +(2)];

short int min1 = 0;
short int min2 = 0;
short int agr = 1; //aggregate sign

if(input1 < 0){
agr = agr*(-1);
}
if(input2 < 0){
agr = agr*(-1);
}
if(input3 < 0){
agr = agr*(-1);
}

//Check first two inputs to get initial min1 and min2
if(abs(input1) <= abs(input2)){
min1 = input1;
min2 = input2;
}
else{
min1 = input2;
min2 = input1;
}

//Check input3 against min1 and min2
if(abs(input3) <= abs(min1)){
min2 = min1;
min1 = input3;
}
else
if(abs(input3) <= abs(min2)){
min2 = input3;
}


/*_________________________________Record Results back to Device Memory________________________________________*/
//Write outputs to the same addresses read initially from the global memory to get the input integers

results_array[current_Index + offset] = min2*agr;
results_array[current_Index + offset +(1)] = min1*agr;
results_array[current_Index + offset +(2)] = min1*agr;

}
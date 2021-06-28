#include "includes.h"
__global__ void convolution_kernel(float *output, float *input, float *filter) {
//declare shared memory for this thread block
//the area reserved is equal to the thread block size plus
//the size of the border needed for the computation

//Write a for loop that loads all values needed by this thread block
//from global memory (input) and stores it into shared memory (sh_input)
//that is local to this thread block
//for ( ... ) {
//for ( ... ) {
//...
//}
//}

//synchronize to make all writes visible to all threads within the thread block

//compute using shared memory

//store result in the global memory

//store result to global memory
}
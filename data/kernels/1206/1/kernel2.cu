#include "includes.h"
/*
Problem 1: initialize array of size 32 to 0
Problem 2: change array size to 1024
Problem 3: create another kernel that adds i to array[ i ]
Problem 4: change array size 8000 (check answer to Problem 3 still works)
*/

//initialize array to 0

//add i to array[ i ]

__global__ void kernel2( int N, int *d_array ){
for( int i = 0; i < N; i++ ){
d_array[ i ] = i;
}
}
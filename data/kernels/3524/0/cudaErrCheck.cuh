#include <stdio.h>

/*-----------------------------
        Cuda Error Check
    Throws error on cuda function
    call
-----------------------------*/
static void ErrorCheck( cudaError_t err, const char *file, int line )
{
    if(err != cudaSuccess)
	{
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define ERROR_CHECK( err ) (ErrorCheck( err, __FILE__, __LINE__ ))
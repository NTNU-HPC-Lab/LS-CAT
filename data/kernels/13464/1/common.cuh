#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//#define _MSC_VER

#if defined(_MSC_VER)
    //  Microsoft 
    #define MY_EXPORT __declspec(dllexport)
    #define MY_IMPORT __declspec(dllimport)
#elif defined(_GCC)
    //  GCC
    #define MY_EXPORT __attribute__((visibility("default")))
    #define MY_IMPORT
#else
    //  do nothing and hope for the best?
    #define MY_EXPORT
    #define MY_IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

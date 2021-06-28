#include "includes.h"








extern "C"




extern "C"






extern "C"





extern "C"





extern "C"




extern "C"


//=== Vector arithmetic ======================================================

extern "C"


extern "C"


extern "C"



extern "C"



extern "C"

extern "C"




//=== Vector-and-scalar arithmetic ===========================================

extern "C"


extern "C"


extern "C"


extern "C"




extern "C"


extern "C"


extern "C"


extern "C"









//=== Vector comparison ======================================================

extern "C"


extern "C"


extern "C"


extern "C"


extern "C"



extern "C"




//=== Vector-and-scalar comparison ===========================================

extern "C"


extern "C"


extern "C"


extern "C"


extern "C"


extern "C"











//=== Vector math (one argument) =============================================


// Calculate the arc cosine of the input argument.
extern "C"


// Calculate the nonnegative arc hyperbolic cosine of the input argument.
extern "C"


// Calculate the arc sine of the input argument.
extern "C"


// Calculate the arc hyperbolic sine of the input argument.
extern "C"


// Calculate the arc tangent of the input argument.
extern "C"


// Calculate the arc hyperbolic tangent of the input argument.
extern "C"


// Calculate the cube root of the input argument.
extern "C"


// Calculate ceiling of the input argument.
extern "C"


// Calculate the cosine of the input argument.
extern "C"


// Calculate the hyperbolic cosine of the input argument.
extern "C"


// Calculate the cosine of the input argument × p .
extern "C"


// Calculate the complementary error function of the input argument.
extern "C"


// Calculate the inverse complementary error function of the input argument.
extern "C"


// Calculate the scaled complementary error function of the input argument.
extern "C"


// Calculate the error function of the input argument.
extern "C"


// Calculate the inverse error function of the input argument.
extern "C"


// Calculate the base 10 exponential of the input argument.
extern "C"


// Calculate the base 2 exponential of the input argument.
extern "C"


// Calculate the base e exponential of the input argument.
extern "C"


// Calculate the base e exponential of the input argument, minus 1.
extern "C"


// Calculate the absolute value of its argument.
extern "C"


// Calculate the largest integer less than or equal to x.
extern "C"


// Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
extern "C"


// Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
extern "C"


// Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
extern "C"


// Calculate the base 10 logarithm of the input argument.
extern "C"


// Calculate the value of l o g e ( 1 + x ) .
extern "C"


// Calculate the base 2 logarithm of the input argument.
extern "C"


// Calculate the doubleing point representation of the exponent of the input argument.
extern "C"


// Calculate the natural logarithm of the input argument.
extern "C"


// Calculate the standard normal cumulative distribution function.
extern "C"


// Calculate the inverse of the standard normal cumulative distribution function.
extern "C"


// Calculate reciprocal cube root function.
extern "C"


// Round input to nearest integer value in doubleing-point.
extern "C"


// Round to nearest integer value in doubleing-point.
extern "C"


// Calculate the reciprocal of the square root of the input argument.
extern "C"


// Calculate the sine of the input argument.
extern "C"


// Calculate the hyperbolic sine of the input argument.
extern "C"


// Calculate the sine of the input argument × p .
extern "C"


// Calculate the square root of the input argument.
extern "C"


// Calculate the tangent of the input argument.
extern "C"


// Calculate the hyperbolic tangent of the input argument.
extern "C"


// Calculate the gamma function of the input argument.
extern "C"


// Truncate input argument to the integral part.
extern "C"


// Calculate the value of the Bessel function of the second kind of order 0 for the input argument.
extern "C"


// Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
extern "C"











//=== Vector math (two arguments) ============================================





// Create value with given magnitude, copying sign of second value.
extern "C"

// Compute the positive difference between x and y.
extern "C"

// Divide two doubleing point values.
extern "C"

// Determine the maximum numeric value of the arguments.
extern "C"

// Determine the minimum numeric value of the arguments.
extern "C"

// Calculate the doubleing-point remainder of x / y.
extern "C"

// Calculate the square root of the sum of squares of two arguments.
extern "C"

// Return next representable single-precision doubleing-point value afer argument.
extern "C"

// Calculate the value of first argument to the power of second argument.
extern "C"

// Compute single-precision doubleing-point remainder.
extern "C"















extern "C"









extern "C"



extern "C"





extern "C"




extern "C"




extern "C"




extern "C"



extern "C"



//WARNING : device_sum size should be gridDim.x
__global__ void vec_addPhotonsAndBackgroundManyReshuffle_scmos (int n, int sizeSubImage,int numberPSFperModel,double *output, double *input, double *photonAndBackground, double * scmos)
{


//print("to do as previous function");



int idx = threadIdx.x + blockIdx.x * blockDim.x;
int idy = threadIdx.y + blockIdx.y * blockDim.y;
int id = idy * gridDim.x * blockDim.x + idx;

if (id < n)
{
int idPSF=id/sizeSubImage;
int idModel=idPSF/numberPSFperModel;
int idoffset=id%sizeSubImage;
int idposit=idPSF%numberPSFperModel;
int idreshuffle=idModel*sizeSubImage +idposit*sizeSubImage*(n/(sizeSubImage*numberPSFperModel))       +idoffset;
output[idreshuffle]=input[id]*photonAndBackground[idPSF*2]+photonAndBackground[idPSF*2+1]+scmos[id];

}

}
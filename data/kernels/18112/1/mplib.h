/**
 * \Created on: Mar 18, 2020
 * \Author: Asena Durukan, Asli Altiparmak, Elif Ozbay, Hasan Ozan Sogukpinar, Nuri Furkan Pala
 * \file mplib.h
 * \brief A library to represent multi-precision
 *      integers and do arithmetic on them.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


/**
 * \def W
 * \brief Size of a digit of a multi-precision
 *      integer.
 *
 * The numbers are represented in base \f$2^{W}\f$
 * such that a digit of the number cannot exceed
 * \f$2^{W}\f$.
 */
#define W 32

//TODO: check graphic card's max thread

#define THREADNUM 256
#define BLOCKNUM 1
#define SIZE 256


/**
 * \brief Type definition for unsigned long
 *      pointer
 */
typedef unsigned long *uni;
/**
 * \brief Type definition for unsigned long
 */
typedef unsigned long uni_t;
/**
 * \brief Type definition for unsigned integer
 *      pointer
 */
typedef unsigned int *ui;
/**
 * \brief Type definition for unsigned integer
 */
typedef unsigned int ui_t;

#if defined(_MSC_VER)
#  define ASM asm volatile
#else
#  define ASM asm __volatile__
#endif

/**
 * \brief Addition without carry
 * @param[out] r result of the addition
 * @param[in] a first number for addition
 * @param[in] b second number for addition

 */
#define __add_cc(r,a,b) ASM ("add.cc.u32 %0, %1, %2;": "=r"(r): "r"(a), "r"(b))

/**
 * \brief Addition with carry
 * @param[out] r result of the addition
 * @param[in] a first number for addition
 * @param[in] b second number for addition

 */
#define __addc_cc(r,a,b) ASM ("addc.cc.u32 %0, %1, %2;": "=r"(r): "r"(a), "r"(b))

/**
 * \brief Subtraction without carry
 * @param[out] r result of the subtraction (a-b)
 * @param[in] a first number for subtraction
 * @param[in] b second number for subtraction

 */
#define __sub_cc(r,a,b) ASM ("sub.cc.u32 %0, %1, %2;": "=r"(r): "r"(a), "r"(b))

/**
 * \brief Subtraction with carry
 * @param[out] r result of the subtraction (a-b)
 * @param[in] a first number for subtraction
 * @param[in] b second number for subtraction

 */
#define __subc_cc(r,a,b) ASM ("subc.cc.u32 %0, %1, %2;": "=r"(r): "r"(a), "r"(b))

/**
 * \brief Getting carry of the addition
 * @param[out] r carry
 */
#define __addcy(carry) ASM ("addc.u32 %0, 0, 0;": "=r"(carry))

/**
 * \brief Adding carry of the addition
 * @param[out] r r + carry
 */
#define __addcy2(carry) ASM ("addc.cc.u32 %0, %0, 0;": "+r"(carry))

/**
 * \brief Getting carry of the subtraction
 * @param[out] r carry
 */
#define __subcy(carry) ASM ("subc.u32 %0, 0, 0;": "=r"(carry))

/**
 * \brief Adding carry of the subtraction
 * @param[out] r r + carry
 */
#define __subcy2(carry) ASM ("subc.s32 %0, %0, 0;": "+r"(carry))

/**
 * \brief Multiplication - Low Bits
 * @param[out] r least significant 32 bits of the multiplication
 * @param[in] a first number for multiplication
 * @param[in] b second number for multiplication

 */
#define __mul_lo(r,a,b) ASM("mul.lo.u32 %0, %1, %2;": "=r"(r): "r"(a),"r"(b))

/**
 * \brief Multiplication - High Bits
 * @param[out] r most significant 32 bits of the multiplication
 * @param[in] a first number for multiplication
 * @param[in] b second number for multiplication

 */
#define __mul_hi(r,a,b) ASM("mul.hi.u32 %0, %1, %2;": "=r"(r): "r"(a),"r"(b))

/**
 * \brief Copies \f$end - start\f$ elements
 *      from a to z.
 * @param[out] z destination of the copy operation
 * @param[in] a source of the copy operation
 * @param[in] start starting index for the copy operation
 * @param[in] end ending index for the copy operation
 */
#define big_cpy(z, a, start, end) if(1) { \
    int i, j; \
    for(i = 0, j = (start); i < (end); i++, j++) { \
        z[i] = a[j]; \
    } \
};
#define bigCopy(z, startZ, endZ, a, startA) if(1) { \
	int i, j; \
	for(i = (startZ), j = (startA); i < (endZ); i++, j++) { \
		z[i] = a[j]; \
	} \
};

/**
 * \brief Initializes z with a random multi-precision
 *      number
 * @param[out] z multi-precision number to be initialized
 * @param[in] l number of digits of z in base \f$2^W\f$
 */
void big_rand(ui z, ui_t l);

/**
 * \brief Initializes z with a random multi-precision
 *      number mod n
 * @param[out] z multi-precision number to be initialized
 * @param[in] l number of digits of z in base \f$2^W\f$
 * @param[in] n modular base for z
 * @param[in] nl number of digits of n in base \f$2^W\f$
 * @param[in] mu precalculated value of  \f$(2^{W})^{2*nl} / n\f$
 * @param[in] mul number of digits of mu in base \f$2^W\f$
 */
void big_mod_rand(ui z, ui_t l, ui n, ui_t nl, ui mu, ui_t mul);

/**
 * \brief Prints the given multi-precision number in
 *      Magma assignment format
 * @param[in] fp pointer to the file to print
 * @param[in] a multi-precision number to be printed
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] s name of the variable going to be assigned to a
 * @param[in] R name of the ring that a is going to be defined in (optional)
 */
void big_print(FILE *fp, ui a, ui_t al, const char *s, const char *R);

/**
 * \brief Checks if two multi-precision numbers are equal
 * @param[out] z 1 if equal, 0 otw
 * @param[in] a first number
 * @param[in] b second number
 * @param[in] l number of digits of a and b in base \f$2^W\f$
 */
void big_is_equal(int *z, ui a, ui b, ui_t l);

/**
 * \brief Checks if a multi-precision number is equal to given
 *      unsigned integer
 * @param[out] z 1 if equal, 0 otw
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b unsigned int to be compared
 */
void big_is_equal_ui(int *z, ui a, ui_t al, ui_t b);

/**
 * \brief Adds two multi-precision numbers
 * @param[out] z result of the addition
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b second number
 * @param[in] bl number of digits of b in base \f$2^W\f$
 */
void big_add(ui z, ui a, ui_t al, ui b, ui_t bl);

/**
 * \brief Adds two multi-precision numbers in mod n
 * @param[out] z result of the addition
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b second number
 * @param[in] bl number of digits of b in base \f$2^W\f$
 * @param[in] n modular base for the addition
 * @param[in] nl number of digits of n in base \f$2^W\f$
 * @param[in] mu precalculated value of \f$(2^{W})^{2*nl} / n\f$
 * @param[in] mul number of digits of mu in base \f$2^W\f$
 */
void big_mod_add(ui z, ui a, ui_t al, ui b, ui_t bl, ui n, ui_t nl, ui mu, ui_t mul);

/**
 * \brief Subtracts two multi-precision numbers
 * @param[out] z result of the subtraction
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b second number
 * @param[in] bl number of digits of b in base \f$2^W\f$
 */
void big_sub(ui z, int *d, ui a, ui_t al, ui b, ui_t bl);

/**
 * \brief Subtracts two multi-precision numbers in mod n
 * @param[out] z result of the subtraction
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b second number
 * @param[in] bl number of digits of b in base \f$2^W\f$
 * @param[in] n modular base for the subtraction
 * @param[in] nl number of digits of n in base \f$2^W\f$
 */
void big_mod_sub(ui z, ui a, ui_t al, ui b, ui_t bl, ui n, ui_t nl);

/**
 * \brief Multiplies two multi-precision numbers
 * @param[out] z result of the multiplication
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b second number
 * @param[in] bl number of digits of b in base \f$2^W\f$
 */
void big_mul(ui z, ui a, ui_t al, ui b, ui_t bl);

/**
 * \brief Multiplies two multi-precision numbers in mod n
 * @param[out] z result of the multiplication
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b second number
 * @param[in] bl number of digits of b in base \f$2^W\f$
 * @param[in] n modular base for the multiplication
 * @param[in] nl number of digits of n in base \f$2^W\f$
 * @param[in] mu precalculated value of \f$(2^{W})^{2*nl} / n\f$
 * @param[in] mul number of digits of mu in base \f$2^W\f$
 */
void big_mod_mul(ui z, ui a, ui_t al, ui b, ui_t bl, ui n, ui_t nl, ui mu, ui_t mul);

/**
 * \brief Calculates  \f$(2^{W})^{2*nl} / n\f$
 * @param[out] z result of the calculation
 * @param[in] n n in the equation
 * @param[in] nl number of digits of n in base \f$2^W\f$
 */
void big_get_mu(ui z, ui n, ui_t nl);

/**
 * \brief Calculates \f$(A + 2) / 4\f$
 * \param[out] A24 result of \f$(A + 2) / 4\f$ or a factor of n
 * @param[in] A A in the equation
 * @param[in] n n modular base for the calculation
 * @param[in] number of digits of n in base \f$2^W\f$
 * @param[in] mu precalculated value of \f$(2^{W})^{2*nl} / n\f$
 * @param[in] mul number of digits of mu in base \f$2^W\f$
 * @param[in] flag 1 when calculation succeeds, 0 when factor gets found
 */
void big_get_A24(ui A24, ui A, ui n, ui_t nl, ui mu, ui_t mul, int *flag);

/**
 * \brief Calculates m mod n
 * @param[out] z result of the reduction
 * @param[in] m number to be reduced
 * @param[in] ml number of digits of m in base \f$2^W\f$
 * @param[in] n modular base for the reduction
 * @param[in] nl number of digits of n in base \f$2^W\f$
 * @param[in] mu precalculated value of \f$(2^{W})^{2*nl} / n\f$
 * @param[in] mul number of digits of mu in base \f$2^W\f$
 */
void barret_reduction(ui z, ui m, ui_t ml, ui n, ui_t nl, ui mu, ui_t mul);

void big_gcd(ui d, ui_t dl, ui a, ui_t al, ui b, ui_t bl);
int big_invert(ui z, ui a, ui_t al, ui b, ui_t bl);

/**
 * \brief Allocates Memory for GPU Arrays
 * @param[out] deviceArray result of the memory allocation
 * @param[in] arraySize size of device array
 */
void memoryAllocationGPU (ui *deviceArray, ui_t arraySize);

/**
 * \brief Copies \f$end - start\f$ elements
 *      from a to z. (Parallelized)
 * @param[out] z destination of the copy operation
 * @param[in] a source of the copy operation
 * @param[in] start starting index for the copy operation
 * @param[in] end ending index for the copy operation
 */
__device__ void bigCpy(ui z, ui a, ui_t start, ui_t end);

/**
 * \brief Multiplies two multi-precision numbers (Parallelized)
 * @param[out] z result of the multiplication
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b second number
 * @param[in] bl number of digits of b in base \f$2^W\f$
 */
__device__ void bigMul(ui z, ui a, ui_t al, ui b, ui_t bl);

__device__ void bigPrint(ui a, ui_t al, const char *s);
/**
 * \brief Subtracts two multi-precision numbers (Parallelized)
 * @param[out] z result of the subtraction
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b second number
 * @param[in] bl number of digits of b in base \f$2^W\f$
 */
__device__ void bigSub(ui z, ui controlBit, ui a, ui_t al, ui b, ui_t bl);

__device__ void bigAdd(ui z, ui a, ui_t al, ui b, ui_t bl);

/**
 * \brief Random Number Generation(Parallelized)
 * @param[out] RANDOM result of random number generation
 * @param[in] globalState needed states for generating random numbers per threads
 */
__device__ float gpuGenerate (curandState* globalState);

/**
 * \brief Initializes z with a random multi-precision
 *      number mod n (Parallelized)
 * @param[out] z multi-precision number to be initialized
 * @param[in] l number of digits of z in base \f$2^W\f$
 * @param[in] n modular base for z
 * @param[in] nl number of digits of n in base \f$2^W\f$
 * @param[in] mu precalculated value of  \f$(2^{W})^{2*nl} / n\f$
 * @param[in] mul number of digits of mu in base \f$2^W\f$
 * @param[in] globalState needed states for generating random numbers per threads
 * @param[in] seed seed for random number generations
 */
__device__ void bigModRand(ui z, ui_t l, ui n, ui_t nl, ui mu, ui_t mul, curandState* globalState);

/**
 * \brief Multiplies two multi-precision numbers in mod n (Parallelized)
 * @param[out] z result of the multiplication
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b second number
 * @param[in] bl number of digits of b in base \f$2^W\f$
 * @param[in] n modular base for the multiplication
 * @param[in] nl number of digits of n in base \f$2^W\f$
 * @param[in] mu precalculated value of \f$(2^{W})^{2*nl} / n\f$
 * @param[in] mul number of digits of mu in base \f$2^W\f$
 */
__device__ void bigModMul(ui z, ui a, ui_t al, ui b, ui_t bl, ui n, ui_t nl, ui mu, ui_t mul);

/**
 * \brief Adds two multi-precision numbers in mod n
 * @param[out] z result of the addition (Parallelized)
 * @param[in] a first number
 * @param[in] al number of digits of a in base \f$2^W\f$
 * @param[in] b second number
 * @param[in] bl number of digits of b in base \f$2^W\f$
 * @param[in] n modular base for the addition
 * @param[in] nl number of digits of n in base \f$2^W\f$
 * @param[in] mu precalculated value of  \f$(2^{W})^{2*nl} / n\f$
 * @param[in] mul number of digits of mu in base \f$2^W\f$
 */
__device__ void bigModAdd(ui z, ui a, ui_t al, ui b, ui_t bl, ui n, ui_t nl, ui mu, ui_t mul);

__device__ void bigIsEqual(ui z, ui a, ui b, ui_t l);

__device__ void bigIsEqualUi(ui z, ui a, ui_t al, ui_t b);

__device__ void bigIsEqualUiD(ui z, ui a, ui_t al, ui_t b);

__device__ void binaryGCD(ui d, ui a, ui_t al, ui n, ui_t nl);

/**
 * \brief Calculates m mod n (Parallelized)
 * @param[out] z result of the reduction
 * @param[in] m number to be reduced
 * @param[in] ml number of digits of m in base \f$2^W\f$
 * @param[in] n modular base for the reduction
 * @param[in] nl number of digits of n in base \f$2^W\f$
 * @param[in] mu precalculated value of \f$(2^{W})^{2*nl} / n\f$
 * @param[in] mul number of digits of mu in base \f$2^W\f$
 */
__device__ void barretReduction(ui z, ui m, ui_t ml, ui n,ui_t nl, ui mu, ui_t mul);

__device__ void bigModSub(ui z, ui a, ui_t al, ui b, ui_t bl, ui n, ui_t nl);

__device__ void bigInvert(ui z, ui a, ui_t al, ui n, ui_t nl, ui mu, ui_t mul);

__global__ void bigPrintG(ui a, ui_t al, char *s);

__device__ void bigRand(ui z, ui_t l, curandState* globalState);

__device__ void bigIsEqualD(ui z, ui a, ui b, ui_t l);

__device__ void bigGetA24(ui z, ui A, ui n, ui_t nl, ui mu, ui_t mul, int *flag);

//Furki

void myset(long int* A, long int t, long int len);

void myshift(long int* A, long int len);

//void mysar(long int* a);

void divsteps2(long int n, long int t, long int* delta, long int f, long int g, long int* uu, long int* vv, long int* qq, long int* rr);

void myMul(long int* zn, long int u, long int* fn, long int v, long int* gn, long int len);

void ufvg(long int u, long int f, long int v, long int g, long int c, long int* h, long int* l);

void safegcd(ui z, ui f, ui g, ui U, ui V, ui Q, ui R, ui precomp, ui_t len);

void myprint(FILE *file, const char *str, long int* a, long int len);

void trace();

void mycopy(long int *destination, long int *source, long int length);

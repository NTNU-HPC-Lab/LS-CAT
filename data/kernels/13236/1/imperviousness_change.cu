#include "includes.h"
//	INCLUDES

// CUDA

// GIS

/**
* 	PARS
*/
#define 					BLOCK_DIM_small				64
#define 					BLOCK_DIM 					256

static const unsigned int 	threads 					= 512;
bool 						print_intermediate_arrays 	= false;
const char 					*BASE_PATH 					= "/home/giuliano/git/cuda/reduction";

/*
*	kernel labels
*/
const char		*kern_0			= "filter_roi";
const char 		*kern_1 		= "imperviousness_change_histc_sh_4"	;
const char 		*kern_2 		= "imperviousness_change"	;
char			buffer[255];

/*
* 		DEFINE I/O files
*/
// I/–
//const char 		*FIL_ROI 		= "/home/giuliano/git/cuda/reduction/data/ROI.tif";
//const char 		*FIL_BIN1 		= "/home/giuliano/git/cuda/reduction/data/BIN1.tif";
//const char 		*FIL_BIN2 		= "/home/giuliano/git/cuda/reduction/data/BIN2.tif";
const char 		*FIL_ROI 		= "/media/DATI/db-backup/ssgci-data/testing/ssgci_roi.tif";
const char 		*FIL_BIN1 		= "/media/DATI/db-backup/ssgci-data/testing/ssgci_bin.tif";
const char 		*FIL_BIN2 		= "/media/DATI/db-backup/ssgci-data/testing/ssgci_bin2.tif";

// –/O
const char 		*FIL_LTAKE_grid	= "/home/giuliano/git/cuda/reduction/data/LTAKE_map.tif";
const char 		*FIL_LTAKE_count= "/home/giuliano/git/cuda/reduction/data/LTAKE_count.txt";

/*	+++++DEFINEs+++++	*/
__global__ void imperviousness_change( const unsigned char *dev_BIN1, const unsigned char *dev_BIN2, unsigned int WIDTH, unsigned int HEIGHT, int *dev_LTAKE_map )
{
unsigned long int x 	= threadIdx.x;
unsigned long int bdx	= blockDim.x;
unsigned long int bix	= blockIdx.x;
unsigned long int tix	= bdx*bix + x;	// offset

if( tix < WIDTH*HEIGHT ){
dev_LTAKE_map[tix]	= (int)((int)dev_BIN2[tix] - (int)dev_BIN1[tix]);
}
}
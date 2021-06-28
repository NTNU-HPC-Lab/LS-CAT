#include "includes.h"

__device__ void computearray_size(int* block_cntr_array,int *finalsize,int *orig_number_of_char)
{
*finalsize = 0;
for(int i=0;i<*orig_number_of_char;i++)
{
(*finalsize)=(*finalsize) + block_cntr_array[i];
}

}
__device__ int char_huffman_table_gpu[MAX_CHAR][MAX_CHAR-1];  //To write the output from compression in GPU  //char *compressedfile_array=0;  bool *compressedfile_array=0;  bool *finalcompressed_array=0;  // To keep track of how many characters each block wrote  int *block_cntr_array=0; int *block_cntr_array_check=0; int *d_last_byte_padding=0;  int *finalsize=0; int *orig_number_of_char=0; int *huffman_check = (int *)malloc((MAX_CHAR)*(MAX_CHAR-1) *sizeof(int));

bool *d_bool = 0;

bool *h_bool = 0;



__global__ void final_compression(int *block_cntr_array,bool *compressedfile_array,bool *finalcompressed_array,int number_of_char)
//__device__ void final_compression(int *block_cntr_array,bool *compressedfile_array,bool *finalcompressed_array)
{
int index_blocks=blockIdx.x*blockDim.x+threadIdx.x;
int index_file=(blockIdx.x*blockDim.x+threadIdx.x)*255;
int final_index=0;

if(index_blocks < number_of_char)
{
for(int i=0;i<index_blocks;i++)
{
final_index = final_index+ block_cntr_array[i];
}
for(int i=0;i<block_cntr_array[index_blocks];i++)
{
finalcompressed_array[final_index+i]=compressedfile_array[index_file+i];
}

}
}
__global__ void compress_file_gpu(unsigned char *d_input,bool *compressedfile_array,int *char_huffman_table2,int *block_cntr_array,int* d_last_byte_padding,int *finalsize,int *orig_number_of_char,int number_of_char)
{
//int write_counter=0,
int block_counter=0;	//how many bits have been written in specific byte
unsigned char input_char;
//unsigned char output_char = 0x0;
//unsigned char end_of_file = 255;
//unsigned char mask = 0x01; //00000001;
int index_file=(blockIdx.x*blockDim.x+threadIdx.x)*255;
int index_blocks=blockIdx.x*blockDim.x+threadIdx.x;

if(index_blocks < number_of_char)
{
//for(int i=0;i<MAX_CHAR;i++)
//{
//int *row = (int*)((char*)char_huffman_table2 + i * pitch);
//for (int c = 0; c < MAX_CHAR-1; ++c) {
//   char_huffman_table_gpu[i][c] = row[c];
//}
//}

input_char = d_input[index_blocks];
for(int i = 0 ; i < (MAX_CHAR - 1) ; i++)
{
if(char_huffman_table2[input_char*255+i] == 0)			//detect if current character on particular position has 0 or 1
{
//output_char = output_char << 1;					//if 0 then shift bits one position to left (last bit after shifting is 0)
compressedfile_array[index_file+i] = false;
//write_counter++;
block_counter++;
}
else if(char_huffman_table2[input_char*255+i] == 1)
{
//output_char = output_char << 1;					//if 1 then shift bits one position to left...
//output_char = output_char | mask;				//...and last bit change to: 1
//write_counter++;
compressedfile_array[index_file+i] = true;
block_counter++;
}
else //-1
{
/*if(input_char == end_of_file)					//if EOF is detected then write current result to file
{
if(write_counter != 0)
{
output_char = output_char << (8-write_counter);
compressedfile_array[index_file]=output_char;
output_char = 0x0;
}
else	//write_counter == 0
{
compressedfile_array[index_file]=output_char;
}
}*/

break;
}

/*if(write_counter == 8)								//if result achieved 8 (size of char) then write it to compressed_file
{
compressedfile_array[index_file]=output_char;
output_char = 0x0;
write_counter = 0;
}*/
}

block_cntr_array[index_blocks]=block_counter;
//*d_last_byte_padding = write_counter;							//to decompress file we have to know how many bits in last byte have been written
//update_config(write_counter); //TODO to zakomentowac przy ostatecznych pomiarach

computearray_size(block_cntr_array,finalsize,orig_number_of_char);
//final_compression(block_cntr_array,compressedfile_array,finalcompressed_array);

}

}
#include "includes.h"
__global__ void build_sequence_length_padding_offset(const int* sequence_length, const int batch_size, const int max_seq_len, int* valid_word_num, int* tmp_mask_offset)
{
// do cumulated sum
int total_seq_len = 0;
int cum_offset = 0;
int index = 0;
for(int i = 0; i < batch_size; i++)
{
const int seq_len = sequence_length[i];
for(int j = 0; j < seq_len; j++)
{
tmp_mask_offset[index] = cum_offset;
index++;
}
cum_offset += max_seq_len - seq_len;
total_seq_len += seq_len;
}
valid_word_num[0] = total_seq_len;
}
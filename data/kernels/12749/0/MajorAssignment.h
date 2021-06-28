#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NO_STEPS 10


// Declare functions
void Allocate_Memory();
void Load_Dat_To_Array(char *input_file_name);
void Init();
void CPUHeatContactFunction();
void CalRenewResult();
#ifdef GPU
void Call_GPUHeatContactFunction();
void Call_GPUTimeStepFunction();
void Send_To_Device();
void Get_From_Device();
#endif
void Free_Memory();
void Save_Data_To_File(char *output_file_name);

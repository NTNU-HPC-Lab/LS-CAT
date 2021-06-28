

class main_skeleton():
    def __init__(self):
        super().__init__()
        self.includes=["#include <stdbool.h>",
                "#include <stdio.h>",
                "#include <string.h>",
                "#include <getopt.h>",
                "#include <curand_kernel.h>",
                "#include <stdlib.h>",
                "#include <cuda.h>",
                "#include <sys/time.h>"]

        self.statics=["#include<chrono>",
                "#include<iostream>",
                "using namespace std;",
                "using namespace std::chrono;",
                "int blocks_[20][2] = {{8,8},{16,16},{24,24},{32,32},{1,64},{1,128},{1,192},{1,256},{1,320},{1,384},{1,448},{1,512},{1,576},{1,640},{1,704},{1,768},{1,832},{1,896},{1,960},{1,1024}};",
                "int matrices_[7][2] = {{240,240},{496,496},{784,784},{1016,1016},{1232,1232},{1680,1680},{2024,2024}};"]
        self.main = ["int main(int argc, char **argv) {","cudaSetDevice(0); ",
                "char* p;int matrix_len=strtol(argv[1], &p, 10);",
                "for(int matrix_looper=0;matrix_looper<matrix_len;matrix_looper++){",
                "for(int block_looper=0;block_looper<20;block_looper++){",
                "int XSIZE=matrices_[matrix_looper][0],YSIZE=matrices_[matrix_looper][1],BLOCKX=blocks_[block_looper][0],BLOCKY=blocks_[block_looper][1];"]

        self.variables = []

        self.thread_dims = ["int iXSIZE=XSIZE;","int iYSIZE=YSIZE;","while(iXSIZE%BLOCKX!=0)","{","iXSIZE++;","}","while(iYSIZE%BLOCKY!=0)","{"," iYSIZE++;","}","dim3 gridBlock(iXSIZE/BLOCKX, iYSIZE/BLOCKY);","dim3 threadBlock(BLOCKX, BLOCKY);"]


        self.function_call = []

        self.end = ["}","auto end = steady_clock::now();","auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);","cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;"]

        self.variable_names = []

    def change_function(self,function_name:str):
        self.function_call.append("cudaFree(0);")
        variables=",".join(self.variable_names)
        self.function_call.append(function_name+"<<<gridBlock,threadBlock>>>("+variables+");")
        self.function_call.append("cudaDeviceSynchronize();")
        self.function_call.append("for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {")
        self.function_call.append(function_name+"<<<gridBlock,threadBlock>>>("+variables+");")
        self.function_call.append("}")
        self.function_call.append("auto start = steady_clock::now();")
        self.function_call.append("for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {")
        self.function_call.append(function_name+"<<<gridBlock,threadBlock>>>("+variables+");")



    def add_variables(self,function_name:str, variables:list):
        for var in variables:
            v=var[0]
            if v.find('*')!=-1:
                self.variables.append(v+var[1]+" = NULL;")
                self.variable_names.append(var[1])
                self.variables.append("cudaMalloc(&"+var[1]+", XSIZE*YSIZE);")
                #self.end.append("cudaFree("+var[1]+");")
            else:

                self.variable_names.append(var[1])
                self.variables.append(v+" "+var[1]+" =  XSIZE*YSIZE;")


    def add_includes(self, includes:list):
        self.includes+=includes

    def save_main(self,loc:str):
        total = self.includes+self.statics+self.main+self.variables+self.thread_dims+self.function_call+self.end+["}","}"+"}"]
        f=open(loc+"/"+'main.cu','w')
        s1='\n'.join(total)
        f.write(s1)
        f.close()

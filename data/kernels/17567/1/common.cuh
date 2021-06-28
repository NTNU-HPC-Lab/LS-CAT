#ifndef COMMON_H
#define COMMON_H

    #include <stdio.h>
    #include <stdlib.h>
    #include <string>

    void CheckGpuPanic () {
        cudaError_t stat = cudaGetLastError();
        if (stat != cudaSuccess) {
            fprintf(stderr, "Error, %s at line %d in file %s\n", cudaGetErrorString(stat), (__LINE__), (__FILE__));
            exit(1);
        }
    }

    void GetGPUGridConfig (dim3 &grid, dim3 &block)
    {
        //Get the device properties
        static bool flag = 0;
        static dim3 lgrid, lthreads;
        if (!flag) {
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, 0);

            //Adjust the grid dimensions based on the device properties
            int num_blocks = 1024 * 2 * devProp.multiProcessorCount;
            lgrid = dim3(num_blocks);
            lthreads = dim3(devProp.maxThreadsPerBlock / 4);
            flag = 1;
        }
        grid = lgrid;
        block = lthreads;
    }

    //dim3 grid, block;
    //get_grid_config(grid, block);

#endif
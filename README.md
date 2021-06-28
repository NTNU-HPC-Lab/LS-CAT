# Running the kernels
Run the runner.sh script in an environment with pandas and numpy,
e.g. in a nvidia-docker container:

`nvidia-docker run -it --rm -v $(pwd):/workspace nvcr.io/nvidia/pytorch:21.06-py3`

The runner.sh script contains four arguments for our python runner.
1. The device ID of the GPU to use for the runs
2. Timeout threshold in seconds for kernels (will terminate the kernel if this is exceeded) 
3. The maximum matrix size. Decrease this if you run out of memory on your GPU.
4. Any nvcc flags as a string, except for -o (already computed based on device ID).

# Results archive
The results archive are completed runs for a GPU architecture that can be used to compare the performance of different models.

## Nvidia T4
The results archive currently contains our T4 results compiled using the default nvcc compilation target of sm_52. The results were created by splitting the kernels into three chunks and running these kernels in parallel on 6 GPUs for each chunk (for a total of 18 T4 GPUs). The results can then be used to analyze variance between GPUs and average out the means.


# Kernel stats
20 259 kernels

Up to 7 matrix sizes
20 block sizes

2 836 260 data points (including any N/A results)
N/A results are dropped at the end of the measurement process.

## Kernel stats T4
Some runs include multiple samples of each kernel, e.g. the T4 dataset contains 2 933 905 non-N/A data points.

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

extern "C" {
	__global__ void base()
	{

		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int bx = blockIdx.x;
		int by = blockIdx.y;

		int i = ((by * blockDim.y + ty) * gridDim.x * blockDim.x) + (bx * blockDim.x + tx);
		//doesn't actually do anything
	}
}

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

extern "C" {
//	__global__ void rng(int *I, int seed)
	__global__ void rng(int *I, int seed)
	{

		//int seed = 1; //TEMP for the time being to make my life easier
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int bx = blockIdx.x;
		int by = blockIdx.y;

		int i = ((by * blockDim.y + ty) * gridDim.x * blockDim.x) + (bx * blockDim.x + tx);


		int delta = 0x9E3779B9;
		int k0 = 0xA341316C;
		int k1 = 0xC8013EA4;
		int k2 = 0xAD90777D;
		int k3 = 0x7E95761E;
		int ITER = 15;

		int x = seed;
		int y = seed << 3;

		x += i + (i << 11) + (i << 19);
		y += i + (i << 9) + (i << 21);    

		int sum = 0;
		for (int j=0; j < ITER; j++) {
			sum += delta;
			x += ((y << 4) + k0) & (y + sum) & ((y >> 5) + k1);
			y += ((x << 4) + k2) & (x + sum) & ((x >> 5) + k3);
		}

		int r = x & 0xFF;
		int g = (x & 0xFF00) >> 8;

		I[i*4  ] = r;
		I[i*4+1] = r;
		I[i*4+2] = r;
		I[i*4+3] = g;
		/*
			 I[i*4  ] = i;
			 I[i*4+1] = i;
			 I[i*4+2] = i;
			 I[i*4+3] = i;
		 */
	}
}

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

__global__ void
rng(int *I, int seed)
{

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

int main(void)
{
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the image size to be used, and compute its size in terms of pixels
  int seed = 1;
  //int width = 640;
  //int height = 480;
  int width = 8192;
  int height = 4096;
  int numElements = height * width;
  size_t numPixels = 4 * numElements * sizeof(int);
  dim3 blocks, threads;

  printf("[Random number generation of a %dx%d image]\n", height, width);

  // Allocate the host output image
  int *h_I = (int *)malloc(numPixels);

  // Verify that allocations succeeded
  if (h_I == NULL)
  {
    fprintf(stderr, "Failed to allocate host image!\n");
    exit(EXIT_FAILURE);
  }


  // Allocate the device input image 
  int *d_I = NULL;
  err = cudaMalloc((void **)&d_I, numPixels);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device image (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel

  blocks = dim3(40,30);
  threads = dim3(16,16);
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocks.x*blocks.y, threads.x*threads.y);
  rng<<< blocks, threads >>>(d_I, seed);
  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch rng kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_I, d_I, numPixels, cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy image data from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result image is correct
  int *t_I = (int *)malloc(numPixels);

  int delta = 0x9E3779B9;
  int k0 = 0xA341316C;
  int k1 = 0xC8013EA4;
  int k2 = 0xAD90777D;
  int k3 = 0x7E95761E;
  int ITER = 15;

  for(int i = 0; i < numElements; i++){

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

    t_I[i*4  ] = r;
    t_I[i*4+1] = r;
    t_I[i*4+2] = r;
    t_I[i*4+3] = g;

  }

  for (int i = 0; i < numElements*4; i++)
  {
    //printf("%d: %d %d\n",i,t_I[i],h_I[i]); 
    //if (fabs(t_I[i] - h_I[i]) > 1e-5)
    if (fabs(t_I[i] - h_I[i]) > 1e-4)
    {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }
  printf("Test PASSED\n");

  // Free device global memory
  err = cudaFree(d_I);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device image (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
  free(h_I);

  // Reset the device and exit
  err = cudaDeviceReset();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return 0;

}


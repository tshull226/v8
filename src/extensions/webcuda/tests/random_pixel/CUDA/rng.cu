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

int main(int argc, char **argv)
{
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  // Print the image size to be used, and compute its size in terms of pixels
  int seed = 1;
  int width = 8192;
  int height = 4096;
  //int width = atoi(argv[1]);
  //int height = atoi(argv[2]);
  int numElements = height * width;
  size_t numPixels = 4 * numElements * sizeof(int);
  dim3 blocks, threads;

  float memcpyHtoD = 0.0;
  float hostMemAlloc, deviceMemAlloc, memcpyDtoH, kernel, hostMemFree, deviceMemFree;
  printf("[Random number generation of a %dx%d image]\n", height, width);

  // Allocate the host output image
    cudaEvent_t start_event, stop_event;
    int eventflags = cudaEventDefault;
    
    cudaEventCreateWithFlags(&start_event, eventflags);
    cudaEventCreateWithFlags(&stop_event, eventflags);

    cudaEventRecord(start_event, 0);  


  int *h_I = (int *)malloc(numPixels);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);  
    cudaEventElapsedTime(&hostMemAlloc, start_event, stop_event);

  // Verify that allocations succeeded
  if (h_I == NULL)
  {
    fprintf(stderr, "Failed to allocate host image!\n");
    exit(EXIT_FAILURE);
  }


  // Allocate the device input image 
  int *d_I = NULL;

  cudaEventRecord(start_event, 0); 
  err = cudaMalloc((void **)&d_I, numPixels);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event); 
    cudaEventElapsedTime(&deviceMemAlloc, start_event, stop_event);


  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device image (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel

  threads = dim3(16,16);
  blocks = dim3(width/threads.x,height/threads.y);

  printf("CUDA kernel launch with %d blocks of %d threads\n", blocks.x*blocks.y, threads.x*threads.y);

    cudaEventRecord(start_event, 0);   
  rng<<< blocks, threads >>>(d_I, seed);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event); 
    cudaEventElapsedTime(&kernel, start_event, stop_event);

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch rng kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy output data from the CUDA device to the host memory\n");
   cudaEventRecord(start_event, 0);
  err = cudaMemcpy(h_I, d_I, numPixels, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event); 
    cudaEventElapsedTime(&memcpyDtoH, start_event, stop_event);


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
    cudaEventRecord(start_event, 0); 
  err = cudaFree(d_I);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   
    cudaEventElapsedTime(&deviceMemFree, start_event, stop_event);


  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device image (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free host memory
    cudaEventRecord(start_event, 0); 
  free(h_I);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&hostMemFree, start_event, stop_event);


    printf("Host Mem Alloc: %f\nDevice Mem Alloc: %f\nMem Copy H to D: %f\nKernel: %f\nMem Copy D to H: %f\nHost Mem Free: %f\nDevice Mem Free: %f\n\n",
            hostMemAlloc*1000, deviceMemAlloc*1000,memcpyHtoD*1000, kernel*1000,memcpyDtoH*1000, hostMemFree*1000, deviceMemFree*1000);

  // Reset the device and exit
  err = cudaDeviceReset();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return 0;

}


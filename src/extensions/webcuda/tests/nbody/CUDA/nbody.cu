
//TODO PROBABLY HAVE TO DECLARED SHARED MEMORY UP HERE...

__device__ float3
bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
	float3 r;

	// r_ij  [3 FLOPS]
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;

	// distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;

	// invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
	float distSixth = distSqr * distSqr * distSqr;
	float invDistCube =  1.0f/sqrtf(distSixth);

	// s = m_j * invDistCube [1 FLOP]
	float s = bj.w * invDistCube;

	// a_i =  a_i + s * r_ij [6 FLOPS]
	ai.x += r.x * s;
	ai.y += r.y * s;
	ai.z += r.z * s;

	return ai;
}

__device__ float3
tile_calculation(float4 myPosition, float3 accel)
{
	int i;
	extern __shared__ float4[] sharedPos;
	for (i = 0; i < blockDim.x; i++) {
		accel = bodyBodyInteraction(myPosition, sharedPos[i], accel);
	}
	return accel;
}

__global__ void
calculate_forces(void *devX, void *devA)
{
    //typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();
	extern __shared__ float4[] sharedPos;

	float4 *globalX = (float4 *)devX;
	float4 *globalA = (float4 *)devA;
	float4 myPosition;
	int i, tile;
    float3  acc = {0.0f, 0.0f, 0.0f};

		int gtid = blockIdx.x * blockDim.x + threadIdx.x

    for (i = 0, tile = 0; i < N; i += p , tile++)
    {
			int idx = tile * blockDim.x + threadIdx.x;
        sharedPos[threadIdx.x] = positions[idx];

        __syncthreads();

        // This is the "tile_calculation" from the GPUG3 article.
				acc = tile_calculation(myPosition, acc);

        __syncthreads();
    }

    return acc;
}

void check_CUDA_op(cudaError_t error, char * message)
{
  if (err != cudaSuccess)
  {
		fprintf(stderr, message);
    fprintf(stderr, " (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

}
int main(void)
{
  // Error code to check return values for CUDA calls
  cudaError_t err = cudaSuccess;

  dim3 blocks, threads;
	int numPixels = 400; //TEMP


  // Allocate the host output image
  int *h_I = (int *)malloc(numPixels);

  // Verify that allocations succeeded
  if (h_I == NULL)
  {
    fprintf(stderr, "Failed to allocate host image!\n");
    exit(EXIT_FAILURE);
  }


  // Allocate the device input image 
  int *d_X = NULL;
  err = cudaMalloc((void **)&d_X, numPixels);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device image (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int *d_A = NULL;
  err = cudaMalloc((void **)&d_A, numPixels);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device image (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch the Vector Add CUDA Kernel

  blocks = dim3(40);
  threads = dim3(16);
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocks.x, threads.x);
  calculate_forces<<< blocks, threads >>>(d_X, d_A);
  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch rng kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_I, d_X, numPixels, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to copy image data from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free device global memory
  err = cudaFree(d_X);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to free device image (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaFree(d_A);
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




#include <stdio.h>

#include <cuda_runtime.h>

#define NUMBODIES 1024
#define TIMESTEP 0.01 
#define NUMITERATIONS 10 
#define NUMTHREADS 64
#define NUMBLOCKS 16 

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
	//need this for the case of comparing the point to itself
	if(distSqr == 0){
		return ai;
	}

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
	extern __shared__ float4 sharedPos[];
	for (i = 0; i < blockDim.x; i++) {
		accel = bodyBodyInteraction(myPosition, sharedPos[i], accel);
	}
	return accel;
}

	__global__ void
calculate_forces(void *devX, void *devV, int num_bodies, int num_iterations, float timestep)
{
	//typename vec4<T>::Type *sharedPos = SharedMemory<typename vec4<T>::Type>();
	extern __shared__ float4 sharedPos[];

	float4 *globalX = (float4 *)devX;
	float3 *globalV = (float3 *)devV;
	float4 myPosition;
	//overriding for time being
	int i, j,  tile;

	int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	
	for(i = 0; i < num_iterations; i++){
		//have to reset acceleration before each iteration
		float3 acc = {0.0f, 0.0f, 0.0f};

		myPosition = globalX[gtid];
		//calculating new acceleration
		for (j = 0, tile = 0; j < num_bodies; j += blockDim.x , tile++)
		{
			int idx = tile * blockDim.x + threadIdx.x;
			sharedPos[threadIdx.x] = globalX[idx];

			//cannot use shared memory before all threads have put in proper value
			__syncthreads();

			// This is the "tile_calculation" from the GPUG3 article.
			acc = tile_calculation(myPosition, acc);

			__syncthreads();
		}
		//modifying velocity
		globalV[gtid].x += timestep*acc.x;
		globalV[gtid].y += timestep*acc.y;
		globalV[gtid].z += timestep*acc.z;

		//modifying position
		globalX[gtid].x += timestep*globalV[gtid].x;
		globalX[gtid].y += timestep*globalV[gtid].y;
		globalX[gtid].z += timestep*globalV[gtid].z;
		//need to wait for all thread to calculate new result before continuing
		__syncthreads();
	}


}

void check_CUDA_op(cudaError_t error, char * message)
{
	if (error != cudaSuccess)
	{
		fprintf(stderr, "%s", message);
		fprintf(stderr, " (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

}

void read_file(char *pathname, int length, float4 *position, float3 *velocity){
	FILE *fp;
	fp = fopen(pathname, "r");
	int i = 0;
	while(fscanf(fp, "%f %f %f %f %f %f %f", 
				&(position[i].w),
				&(position[i].x),
				&(position[i].y),
				&(position[i].z),
				&(velocity[i].x),
				&(velocity[i].y),
				&(velocity[i].z)) != EOF){
		i++;
	}
	printf("number of rows found: %d\n", i);

	fclose(fp);
}

void write_file(char *pathname, int length, float4 *position, float3 *velocity){
	FILE *fp;
	fp = fopen(pathname, "w");
	for(int i=0; i < length; i++){
		fprintf(fp, "%f %f %f %f",
				position[i].w, position[i].x, position[i].y, position[i].z);
		fprintf(fp, " %f %f %f\n",
				velocity[i].x, velocity[i].y, velocity[i].z);
	}
	printf("done writing file\n");

	fclose(fp);
}

int main(void)
{
	dim3 blocks, threads;

	char * pathname = "../data/tab1024";

	//allocate host memory for position, velocity
	float4 *h_X = (float4 *) malloc(sizeof(float4) * NUMBODIES);
	float3 *h_V = (float3 *) malloc(sizeof(float3) * NUMBODIES);

	read_file(pathname, NUMBODIES, h_X, h_V);


	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;


	// Allocate the device input image 
	float4 *d_X = NULL;
	err = cudaMalloc((void **)&d_X, NUMBODIES*sizeof(float4));
	check_CUDA_op(err, "Failed to allocate device memory for position array");

	float3 *d_V = NULL;
	err = cudaMalloc((void **)&d_V, NUMBODIES*sizeof(float3));
	check_CUDA_op(err, "Failed to allocate device memory for vector array");

	//Copy position, vector data over to kernel
	printf("Copy position data from the CUDA device to the host memory\n");
	err = cudaMemcpy(d_X, h_X, NUMBODIES*sizeof(float4), cudaMemcpyHostToDevice);
	check_CUDA_op(err, "Failed to copy position data to device");

	printf("Copy vector data from the CUDA device to the host memory\n");
	err = cudaMemcpy(d_V, h_V, NUMBODIES*sizeof(float3), cudaMemcpyHostToDevice);
	check_CUDA_op(err, "Failed to copy vector data to device");


	int shared_mem_size	= sizeof(float4)*NUMTHREADS;

	printf("shared mem size %d\n", shared_mem_size);
	// Launch the Vector Add CUDA Kernel
	blocks = dim3(NUMBLOCKS);
	threads = dim3(NUMTHREADS);
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocks.x, threads.x);
	calculate_forces<<< blocks, threads, shared_mem_size >>>(d_X, d_V, NUMBODIES, NUMITERATIONS, TIMESTEP);
	err = cudaGetLastError();
	check_CUDA_op(err, "Failed to to launch calculate_forces kernel");


	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_X, d_X, NUMBODIES*sizeof(float4), cudaMemcpyDeviceToHost);
	check_CUDA_op(err, "Failed to to launch calculate_forces kernel");

	// Free device global memory
	err = cudaFree(d_X);
	check_CUDA_op(err, "Failed to free device position memory");

	err = cudaFree(d_V);
	check_CUDA_op(err, "Failed to free device velocity memory");

	//write results
	//write_file("temp.txt", NUMBODIES, h_X, h_V);

	// Free host memory
	free(h_X);
	free(h_V);

	// Reset the device and exit
	err = cudaDeviceReset();
	check_CUDA_op(err, "Failed to deinitialize the device ");

	return 0;

}




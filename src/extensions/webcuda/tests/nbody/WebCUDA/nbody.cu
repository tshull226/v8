
#include <stdio.h>

#include <cuda_runtime.h>

extern "C" {

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
}



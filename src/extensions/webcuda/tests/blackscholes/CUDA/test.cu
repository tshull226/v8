/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

#include <stdio.h>
#include "bs_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    //float t = 0.5f;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//const int OPT_N = 4000000;
//const int NUM_ITERATIONS = 512;
const int OPT_N = 1024;
const int NUM_ITERATIONS = 1;


const int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    //'h_' prefix - CPU (host) memory space
    float
    //CPU copy of GPU results
    *h_CallResultGPU,
    *h_PutResultGPU,
    //CPU instance of input data
    *h_StockPrice,
    *h_OptionStrike,
    *h_OptionYears;

    //'d_' prefix - GPU (device) memory space
    float
    //Results calculated by GPU
    *d_CallResult,
    *d_PutResult,
    //GPU instance of input data
    *d_StockPrice,
    *d_OptionStrike,
    *d_OptionYears;

    float hostMemAlloc, deviceMemAlloc, memcpyHtoD, memcpyDtoH, kernel, hostMemFree, deviceMemFree;

    printf("Initializing data...\n");
    printf("...allocating CPU memory for options.\n");


    cudaEvent_t start_event, stop_event;
    int eventflags = cudaEventDefault;
    
    cudaEventCreateWithFlags(&start_event, eventflags);
    cudaEventCreateWithFlags(&stop_event, eventflags);

    cudaEventRecord(start_event, 0);  

    h_CallResultGPU = (float *)malloc(OPT_SZ);
    h_PutResultGPU  = (float *)malloc(OPT_SZ);
    h_StockPrice    = (float *)malloc(OPT_SZ);
    h_OptionStrike  = (float *)malloc(OPT_SZ);
    h_OptionYears   = (float *)malloc(OPT_SZ);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);  
    cudaEventElapsedTime(&hostMemAlloc, start_event, stop_event);

    printf("...allocating GPU memory for options.\n");

    cudaEventRecord(start_event, 0); 

    cudaMalloc((void **)&d_CallResult,   OPT_SZ);
    cudaMalloc((void **)&d_PutResult,    OPT_SZ);
    cudaMalloc((void **)&d_StockPrice,   OPT_SZ);
    cudaMalloc((void **)&d_OptionStrike, OPT_SZ);
    cudaMalloc((void **)&d_OptionYears,  OPT_SZ);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event); 
    cudaEventElapsedTime(&deviceMemAlloc, start_event, stop_event);

    printf("...generating input data in CPU mem.\n");
    srand(5347);

    //Generate options set
    int i = 0;
    for (i = 0; i < OPT_N; i++)
    {
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }
/*
    for (i = 0; i < OPT_N; i++)
    {
        //printf("%f %f %f\n", h_StockPrice[i], h_OptionStrike[i], h_OptionYears[i]);
        printf("%f, ", h_OptionYears[i]);
    }
*/
    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing

    cudaEventRecord(start_event, 0);   

    cudaMemcpy(d_StockPrice,  h_StockPrice,   OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionStrike, h_OptionStrike,  OPT_SZ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_OptionYears,  h_OptionYears,   OPT_SZ, cudaMemcpyHostToDevice);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   
    cudaEventElapsedTime(&memcpyHtoD, start_event, stop_event);

    printf("Data init done.\n\n");


    printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
    cudaDeviceSynchronize();

    cudaEventRecord(start_event, 0);     

    for (i = 0; i < NUM_ITERATIONS; i++)
    {
        BlackScholesGPU<<<DIV_UP(OPT_N, 128), 128/*480, 128*/>>>(
            d_CallResult,
            d_PutResult,
            d_StockPrice,
            d_OptionStrike,
            d_OptionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N
        );
        //getLastCudaError("BlackScholesGPU() execution failed\n");
    }

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event); 
    cudaEventElapsedTime(&kernel, start_event, stop_event);

    cudaDeviceSynchronize();

    printf("\nReading back GPU results...\n");
    //Read back GPU results to compare them to CPU results
   cudaEventRecord(start_event, 0);

    cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PutResultGPU,  d_PutResult,  OPT_SZ, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event); 
    cudaEventElapsedTime(&memcpyDtoH, start_event, stop_event);

    for (i = 0; i < OPT_N; i++)
    {
    //    printf("%f %f\n", h_CallResultGPU[i], h_PutResultGPU[i]);
    }

    printf("Shutting down...\n");
    printf("...releasing GPU memory.\n");

    cudaEventRecord(start_event, 0); 

    cudaFree(d_OptionYears);
    cudaFree(d_OptionStrike);
    cudaFree(d_StockPrice);
    cudaFree(d_PutResult);
    cudaFree(d_CallResult);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&deviceMemFree, start_event, stop_event);

    printf("...releasing CPU memory.\n");

    cudaEventRecord(start_event, 0); 

    free(h_OptionYears);
    free(h_OptionStrike);
    free(h_StockPrice);
    free(h_PutResultGPU);
    free(h_CallResultGPU);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&hostMemFree, start_event, stop_event);

    printf("Shutdown done.\n");

    printf("Host Mem Alloc: %f\nDevice Mem Alloc: %f\nMem Copy H to D: %f\nKernel: %f\nMem Copy D to H: %f\nHost Mem Free: %f\nDevice Mem Free: %f\n\n",
            hostMemAlloc*1000000, deviceMemAlloc*1000000,memcpyHtoD*1000000, kernel*1000000,memcpyDtoH*1000000, hostMemFree*1000000, deviceMemFree*1000000);

    cudaDeviceReset();

    //exit(EXIT_SUCCESS);
    return 0;
}

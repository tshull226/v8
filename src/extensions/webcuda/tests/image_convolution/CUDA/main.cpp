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
* This sample implements a separable convolution filter
* of a 2D image with an arbitrary kernel.
*/

// CUDA runtime
#include "cuda_runtime.h"

// Utilities and system includes
#include "helper_functions.h"
#include "helper_cuda.h"

#include "convolutionSeparable_common.h"

////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void convolutionColumnCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // start logs
    printf("[%s] - Starting...\n", argv[0]);

    float
    *h_Kernel,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;

    float
    *d_Input,
    *d_Output,
    *d_Buffer;

    float hostMemAlloc, deviceMemAlloc, findDev,  memcpyHtoD, memcpyDtoH, kernel, hostMemFree, deviceMemFree;

    const int imageW = 4096;
    const int imageH = 4096;
    const int iterations = 0;

    StopWatchInterface *hTimer = NULL;

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    cudaEvent_t start_event, stop_event;
    int eventflags = cudaEventDefault;
    
    cudaEventCreateWithFlags(&start_event, eventflags);
    cudaEventCreateWithFlags(&stop_event, eventflags);

    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed
    findCudaDevice(argc, (const char **)argv);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&findDev, start_event, stop_event);


    sdkCreateTimer(&hTimer);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");

    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

    h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&hostMemAlloc, start_event, stop_event);

    srand(200);

    for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
    {
        h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        h_Input[i] = (float)(rand() % 16);
    }

    printf("Allocating and initializing CUDA arrays...\n");


    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

    cudaMalloc((void **)&d_Input,   imageW * imageH * sizeof(float));
    cudaMalloc((void **)&d_Output,  imageW * imageH * sizeof(float));
    cudaMalloc((void **)&d_Buffer , imageW * imageH * sizeof(float));

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&deviceMemAlloc, start_event, stop_event);


    setConvolutionKernel(h_Kernel);


    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&memcpyHtoD, start_event, stop_event);


    printf("Running GPU convolution (%u identical iterations)...\n\n", iterations);

    for (int i = -1; i < iterations; i++)
    {
        //i == -1 -- warmup iteration
        if (i == 0)
        {
            cudaDeviceSynchronize();
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }

    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed
        convolutionRowsGPU(
            d_Buffer,
            d_Input,
            imageW,
            imageH
        );

        convolutionColumnsGPU(
            d_Output,
            d_Buffer,
            imageW,
            imageH
        );

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&kernel, start_event, stop_event);




    }

    cudaDeviceSynchronize();
    sdkStopTimer(&hTimer);
    double gpuTime = 0.001 * sdkGetTimerValue(&hTimer) / (double)iterations;
    printf("convolutionSeparable, Throughput = %.4f MPixels/sec, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n",
           (1.0e-6 * (double)(imageW * imageH)/ gpuTime), gpuTime, (imageW * imageH), 1, 0);

    printf("\nReading back GPU results...\n\n");

    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

    cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&memcpyDtoH, start_event, stop_event);


    printf("Checking the results...\n");
    printf(" ...running convolutionRowCPU()\n");
    convolutionRowCPU(
        h_Buffer,
        h_Input,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

    printf(" ...running convolutionColumnCPU()\n");
    convolutionColumnCPU(
        h_OutputCPU,
        h_Buffer,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

    printf(" ...comparing the results\n");
    double sum = 0, delta = 0;

    for (unsigned i = 0; i < imageW * imageH; i++)
    {
        delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
        sum   += h_OutputCPU[i] * h_OutputCPU[i];
    }
  
    double L2norm = sqrt(delta / sum);
    printf(" ...Relative L2 norm: %E\n\n", L2norm);
    printf("Shutting down...\n");

    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed
    cudaFree(d_Buffer);
    cudaFree(d_Output);
    cudaFree(d_Input);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&deviceMemFree, start_event, stop_event);


    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

    free(h_OutputGPU);
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&hostMemFree, start_event, stop_event);

    printf("Host Mem Alloc: %f\nDevice Mem Alloc: %f\nFind Device: %f\nMem Copy H to D: %f\nKernel: %f\nMem Copy D to H: %f\nHost Mem Free: %f\nDevice Mem Free: %f\n\n",
    hostMemAlloc*1000000, deviceMemAlloc*1000000,findDev*1000000,memcpyHtoD*1000000, kernel*1000000,memcpyDtoH*1000000, hostMemFree*1000000, deviceMemFree*1000000);

    sdkDeleteTimer(&hTimer);

    cudaDeviceReset();

    if (L2norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Test passed\n");
        exit(EXIT_SUCCESS);
    }
}

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
* This sample implements 64-bin histogram calculation
* of arbitrary-sized 8-bit data array
*/

#include <stdio.h>
#include "histogram256.cu"

const int numRuns = 1;

int main(int argc, char **argv)
{
    uchar *h_Data;
    uint  *h_HistogramGPU;
    uint  *h_PartialHistogramsGPU;
    uchar *d_Data;
    uint  *d_Histogram, *d_PartialHistograms;
    //uint byteCount = 64 * 1048576;
    uint byteCount = 512 * 1024 * 1024;

    float hostMemAlloc, deviceMemAlloc, memcpyHtoD, memcpyDtoH, h_kernel, m_kernel, hostMemFree, deviceMemFree, temp;

    h_kernel = 0.0;
    m_kernel = 0.0;
    memcpyDtoH= 0.0;

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");

    cudaEvent_t start_event, stop_event;
    int eventflags = cudaEventDefault;
    
    cudaEventCreateWithFlags(&start_event, eventflags);
    cudaEventCreateWithFlags(&stop_event, eventflags);

    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

    h_Data         = (uchar *)malloc(byteCount);
    h_HistogramGPU = (uint *)malloc(256 * sizeof(uint));
    h_PartialHistogramsGPU = (uint *)malloc(240 * 256 * sizeof(uint));

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&hostMemAlloc, start_event, stop_event);

    printf("...generating input data\n");
    srand(2009);

    for (uint i = 0; i < byteCount; i++)
    {
        h_Data[i] = rand() % 256;
        //printf("%d, ", h_Data[i]);
    }

    printf("...allocating GPU memory and copying input data\n\n");

    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

    cudaMalloc((void **)&d_Data, byteCount);
    cudaMalloc((void **)&d_Histogram, 256 * sizeof(uint));
    cudaMalloc((void **)&d_PartialHistograms, 240 * 256 * sizeof(uint));

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&deviceMemAlloc, start_event, stop_event);


    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

    cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&memcpyHtoD, start_event, stop_event);

    {
        printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);

        for (int iter = 0; iter < numRuns; iter++)
        {
            if (iter == 0)
            {
                cudaDeviceSynchronize();
            }

            cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed
            histogram256Kernel<<<240, 192>>>(
                d_PartialHistograms,
                (uint *)d_Data,
                byteCount / sizeof(uint)
            );

            cudaEventRecord(stop_event, 0);
            cudaEventSynchronize(stop_event);   // block until the event is actually recorded
            cudaEventElapsedTime(&temp, start_event, stop_event);
            h_kernel += temp;
            
            cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

            cudaMemcpy(h_PartialHistogramsGPU, d_PartialHistograms, 240 * 256 * sizeof(uint), cudaMemcpyDeviceToHost);

            cudaEventRecord(stop_event, 0);
            cudaEventSynchronize(stop_event);   // block until the event is actually recorded
            cudaEventElapsedTime(&temp, start_event, stop_event);
            memcpyDtoH += temp;
            
            cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed
            /*for (uint i = 0; i < 240 * 256; i++)
            {
                printf("%d\n", h_PartialHistogramsGPU[i]);
            }*/

            mergeHistogram256Kernel<<<256, 256>>>(
                d_Histogram,
                d_PartialHistograms,
                240
            );

            cudaEventRecord(stop_event, 0);
            cudaEventSynchronize(stop_event);   // block until the event is actually recorded
            cudaEventElapsedTime(&temp, start_event, stop_event);
            m_kernel += temp;
            
        }

        cudaDeviceSynchronize();
        printf(" ...reading back GPU results\n");

        cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

        cudaMemcpy(h_HistogramGPU, d_Histogram, 256 * sizeof(uint), cudaMemcpyDeviceToHost);

        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);   // block until the event is actually recorded
        cudaEventElapsedTime(&temp, start_event, stop_event);
        memcpyDtoH += temp;
 
        printf(" ...printing GPU results\n");
        for (uint i = 0; i < 256; i++)
        {
            //printf("%d\n", h_HistogramGPU[i]);
        }
    }

    printf("Shutting down...\n");

    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

    cudaFree(d_Histogram);
    cudaFree(d_PartialHistograms);
    cudaFree(d_Data);


    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&deviceMemFree, start_event, stop_event);



    cudaEventRecord(start_event, 0);     // record in stream-0, to ensure that all previous CUDA calls have completed

    free(h_HistogramGPU);
    free(h_Data);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);   // block until the event is actually recorded
    cudaEventElapsedTime(&hostMemFree, start_event, stop_event);

    printf("Host Mem Alloc: %f\nDevice Mem Alloc: %f\nMem Copy H to D: %f\nHistogram Kernel: %f\nMerge Kernel: %f\nMem Copy D to H: %f\nHost Mem Free: %f\nDevice Mem Free: %f\n",
            hostMemAlloc*1000, deviceMemAlloc*1000,memcpyHtoD*1000, h_kernel*1000, m_kernel*1000, memcpyDtoH*1000, hostMemFree*1000, deviceMemFree*1000);
    cudaDeviceReset();

    return 0;
}

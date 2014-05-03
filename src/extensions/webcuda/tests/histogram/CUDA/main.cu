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

const int numRuns = 0;

int main(int argc, char **argv)
{
    uchar *h_Data;
    uint  *h_HistogramGPU;
    uint  *h_PartialHistogramsGPU;
    uchar *d_Data;
    uint  *d_Histogram, *d_PartialHistograms;
    //uint byteCount = 64 * 1048576;
    uint byteCount = 64 * 1024;

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    h_Data         = (uchar *)malloc(byteCount);
    h_HistogramGPU = (uint *)malloc(256 * sizeof(uint));
    h_PartialHistogramsGPU = (uint *)malloc(240 * 256 * sizeof(uint));

    printf("...generating input data\n");
    srand(2009);

    for (uint i = 0; i < byteCount; i++)
    {
        h_Data[i] = rand() % 256;
        //printf("%d, ", h_Data[i]);
    }

    printf("...allocating GPU memory and copying input data\n\n");
    cudaMalloc((void **)&d_Data, byteCount);
    cudaMalloc((void **)&d_Histogram, 256 * sizeof(uint));
    cudaMalloc((void **)&d_PartialHistograms, 240 * 256 * sizeof(uint));
    cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice);

    {
        printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);

        for (int iter = -1; iter < numRuns; iter++)
        {
            //iter == -1 -- warmup iteration
            if (iter == 0)
            {
                cudaDeviceSynchronize();
            }

            histogram256Kernel<<<240, 192>>>(
                d_PartialHistograms,
                (uint *)d_Data,
                byteCount / sizeof(uint)
            );

            cudaMemcpy(h_PartialHistogramsGPU, d_PartialHistograms, 240 * 256 * sizeof(uint), cudaMemcpyDeviceToHost);
            /*for (uint i = 0; i < 240 * 256; i++)
            {
                printf("%d\n", h_PartialHistogramsGPU[i]);
            }*/

            mergeHistogram256Kernel<<<256, 256>>>(
                d_Histogram,
                d_PartialHistograms,
                240
            );
        }

        cudaDeviceSynchronize();
        printf(" ...reading back GPU results\n");
        cudaMemcpy(h_HistogramGPU, d_Histogram, 256 * sizeof(uint), cudaMemcpyDeviceToHost);

        printf(" ...printing GPU results\n");
        for (uint i = 0; i < 256; i++)
        {
            printf("%d\n", h_HistogramGPU[i]);
        }
    }

    printf("Shutting down...\n");
    cudaFree(d_Histogram);
    cudaFree(d_PartialHistograms);
    cudaFree(d_Data);
    free(h_HistogramGPU);
    free(h_Data);

    cudaDeviceReset();

    return 0;
}

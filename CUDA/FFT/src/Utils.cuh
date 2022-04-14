#pragma once

// Gets rid of false flags with IntelliSense
#ifdef __CUDACC__
	#define KERNEL_GRID(grid, block) <<< grid, block >>>
#else
	#define KERNEL_GRID(grid, block)
#endif

#define CUDA_TIMER_START() float time; cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0)
#define CUDA_TIMER_END() cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&time, start, stop)
#define CUDA_TIMER_PRINT() printf("Time for the kernel: %f us\n", time * 1000.0f)

void PrintDeviceInfo()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	printf("Number of devices: %d\n", nDevices);

	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (MHz): %d\n",
			prop.memoryClockRate / 1024);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
			2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		printf("  Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
		printf("  Total constant memory (bytes) %d\n", (int)(prop.totalConstMem));
		printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock) / 1024.0);
		printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
		printf("  Warp-size: %d\n", prop.warpSize);
		printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
		printf("  Concurrent computation/communication: %s\n\n", prop.deviceOverlap ? "yes" : "no");
	}
}
#pragma once
#include <stdio.h>

typedef unsigned int uint;

#define asfloat(x) static_cast<float>(x)

// Gets rid of false flags with IntelliSense
#ifdef __CUDACC__
	#define KERNEL_GRID(grid, block) <<< grid, block >>>
#else
	#define KERNEL_GRID(grid, block)
#endif

struct Timer
{
	Timer()
	{
		cudaEventCreate(&_start);
		cudaEventCreate(&_stop);
		cudaEventRecord(_start);
	}
	~Timer()
	{
		cudaEventDestroy(_start);
		cudaEventDestroy(_stop);
	}

	inline float evaluate()
	{
		cudaEventRecord(_stop);
		cudaEventSynchronize(_stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, _start, _stop);
		return milliseconds;
	}
	inline void eval_print()
	{
		printf("%.3f\n", evaluate());
	}

private:
	cudaEvent_t _start, _stop;
};

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
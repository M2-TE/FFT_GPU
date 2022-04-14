#pragma once

typedef unsigned int uint;

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

// almost consistent overhead (?):
// Debug: ~311 clock cycles
// Release: ~6 clock cycles
#define START() clock_t start = clock()
#define STOP() clock_t stop = clock()
#define WRITE_CYCLES(tid) pCycles[tid] = (uint)(stop - start);

__device__ uint ReverseBits1(uint n)
{
	uint32_t x;
	for (uint i = 31u; n; ) {
		x |= (n & 1u) << i;
		n >>= 1u;
		--i;
	}
	return x;
}
__device__ uint ReverseBits2(uint n)
{
	n = (n >> 1u) & 0x55555555 | (n << 1u) & 0xaaaaaaaa;
	n = (n >> 2u) & 0x33333333 | (n << 2u) & 0xcccccccc;
	n = (n >> 4u) & 0x0f0f0f0f | (n << 4u) & 0xf0f0f0f0;
	n = (n >> 8u) & 0x00ff00ff | (n << 8u) & 0xff00ff00;
	n = (n >> 16u) & 0x0000ffff | (n << 16u) & 0xffff0000;
	return n;
}
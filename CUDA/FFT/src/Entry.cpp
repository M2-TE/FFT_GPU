#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#include "Utils.cuh"
//#include "revised/fft_cg_2w.cuh"
//#include "revised/fft_2w.cuh"
#include "revised/fft_cg_r2_2_s2_1024.cuh"
#include "KernelLauncher.cuh"

int main()
{
	ExecuteFFT<1024>();
	//DoA();
}
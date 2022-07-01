// constexpr math library
//#include "gcem/include/gcem.hpp"

// src code
#include "utils.cuh"
#include "fft_data.cuh"
#include "cufft_impl.cuh"
#include "custom_fft_impl.cuh"

int main()
{
	static constexpr uint N = 4096; // N-point fft

	FFTData data, cufftData;
	data.init_a(N).upload();
	cufftData.init_a(N).upload();

	//perform_custom_fft(data);
	perform_cufft(cufftData);

	data.download().print();
	cufftData.download().print();
}
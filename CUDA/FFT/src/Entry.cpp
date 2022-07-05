// constexpr math library
//#include "gcem/include/gcem.hpp"

// src code
#include "utils.cuh"
#include "fft_data.cuh"
#include "cufft_impl.cuh"
#include "custom_fft_impl.cuh"

int main()
{
	static constexpr uint N = 1024; // N-point fft

	FFTData data, cufftData;
	data.init_a(N);
	//data.init_b(N, -1.0f, 1.0f);
	cufftData = data;

	data.upload();
	perform_custom_fft<N>(data);
	data.download();

	cufftData.upload();
	perform_cufft(cufftData);
	cufftData.download();

	data.print();
	//cufftData.print();

	FFTData::compare(data, cufftData);
}
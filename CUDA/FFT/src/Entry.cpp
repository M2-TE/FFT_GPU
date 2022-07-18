// constexpr math library
//#include "gcem/include/gcem.hpp"

// src code
#include "utils.cuh"
#include "fft_data.cuh"
#include "cufft_impl.cuh"
#include "custom_fft_impl.cuh"

int main()
{
	static constexpr uint N = 64; // N-point fft
	FFTData<float> data;
	FFTData<double> cufftData;
	cufftData.init(N, InitType::eGradient);
	data = cufftData;

	// custom fft
	{
		//data.upload();
		//perform_custom_fft<N>(data);
		//data.download();
	}

	// cuFFT
	{
		cufftData.upload();
		//perform_cufft(cufftData);
		perform_cufft_double(cufftData);
		cufftData.download();
	}

	//data.print();
	cufftData.print();

	//FFTData<float>::compare(data, cufftData);
}
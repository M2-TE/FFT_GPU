#include <cufft.h>

template<bool bDebug = false> inline void perform_cufft(FFTData<float>& data)
{
	cufftHandle plan;
	cufftPlan1d(&plan, data.size(), CUFFT_C2C, 1);
	cufftComplex* inOut = reinterpret_cast<cufftComplex*>(data.deviceData);

	if constexpr (bDebug) {
		Timer timer;
		cufftExecC2C(plan, inOut, inOut, CUFFT_FORWARD);
		timer.eval_print();
	} else cufftExecC2C(plan, inOut, inOut, CUFFT_FORWARD);

	cufftDestroy(plan);
}

template<bool bDebug = false> inline void perform_cufft_double(FFTData<double>& data)
{
	cufftHandle plan;
	cufftPlan1d(&plan, data.size(), CUFFT_Z2Z, 1);
	cufftDoubleComplex* inOut = reinterpret_cast<cufftDoubleComplex*>(data.deviceData);

	if constexpr (bDebug) {
		Timer timer;
		cufftExecZ2Z(plan, inOut, inOut, CUFFT_FORWARD);
		timer.eval_print();
	}
	else cufftExecZ2Z(plan, inOut, inOut, CUFFT_FORWARD);

	cufftDestroy(plan);
}
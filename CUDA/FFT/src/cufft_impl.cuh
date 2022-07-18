#include <cufft.h>

void perform_cufft(FFTData<float>& data)
{
	cufftHandle plan;
	cufftPlan1d(&plan, data.size(), CUFFT_C2C, 1);

	cufftComplex* inOut = reinterpret_cast<cufftComplex*>(data.deviceData);
	cufftExecC2C(plan, inOut, inOut, CUFFT_FORWARD);

	cufftDestroy(plan);
}

void perform_cufft_double(FFTData<double>& data)
{
	cufftHandle plan;
	cufftPlan1d(&plan, data.size(), CUFFT_Z2Z, 1);

	cufftDoubleComplex* inOut = reinterpret_cast<cufftDoubleComplex*>(data.deviceData);
	cufftExecZ2Z(plan, inOut, inOut, CUFFT_FORWARD);

	cufftDestroy(plan);
}
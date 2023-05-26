#pragma once

struct real;
struct SIZE_TYPE {

};

class Function
{
public:
	__device__ Function() {}
	__device__ virtual ~Function() {}
	__device__ virtual void Evaluate(const real* __restrict__ positions, real* fitnesses, const SIZE_TYPE particlesCount) const = 0;
};

class FunctionRsj : public Function
{
private:
	SIZE_TYPE m_DimensionsCount;
	SIZE_TYPE m_PointsCount;
	real* m_Y;
	real* m_X;
public:
	__device__ FunctionRsj(const SIZE_TYPE dimensionsCount, const SIZE_TYPE pointsCount, real* configFileData)
		: m_DimensionsCount(dimensionsCount),
		m_PointsCount(pointsCount),
		m_Y(configFileData),
		m_X(configFileData + pointsCount) {}

	__device__ ~FunctionRsj()
	{
		// m_Y points to the beginning of the config
		// file data, use it for destruction as this 
		// object took ownership of configFilDeata.
		delete[] m_Y;
	}

	__device__ void Evaluate(const real* __restrict__ positions, real* fitnesses, const SIZE_TYPE particlesCount) const
	{
		// Implement evaluation of FunctionRsj here.
	}
};

__global__ void evaluate_fitnesses(
	const real* __restrict__ positions,
	real* fitnesses,
	Function const* const* __restrict__ function,
	const SIZE_TYPE particlesCount)
{
	// This whole kernel is just a proxy as kernels
	// cannot be member functions.
	(*function)->Evaluate(positions, fitnesses, particlesCount);
}

__global__ void create_function(
	Function** function,
	SIZE_TYPE dimensionsCount,
	SIZE_TYPE pointsCount,
	real* configFileData)
{
	// It is necessary to create object representing a function
	// directly in global memory of the GPU device for virtual
	// functions to work correctly, i.e. virtual function table
	// HAS to be on GPU as well.
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		(*function) = new FunctionRsj(dimensionsCount, pointsCount, configFileData);
	}
}

__global__ void delete_function(Function** function)
{
	delete* function;
}

int main()
{
	// Lets just assume d_FunctionConfigData, d_Positions,
	// d_Fitnesses are arrays allocated on GPU already ...

	// Create function.
	Function** d_Function;
	cudaMalloc(&d_Function, sizeof(Function**));
	create_function << <1, 1 >> > (d_Function, 10, 10, d_FunctionConfigData);

	// Evaluate using proxy kernel.
	evaluate_fitnesses << <
		m_Configuration.GetEvaluationGridSize(),
		m_Configuration.GetEvaluationBlockSize(),
		m_Configuration.GetEvaluationSharedMemorySize() >> > (
			d_Positions,
			d_Fitnesses,
			d_Function,
			m_Configuration.GetParticlesCount());

	// Delete function object on GPU.
	delete_function << <1, 1 >> > (d_Function);
}
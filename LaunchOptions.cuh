#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct LaunchOptions {
	dim3 threadsPerBlock;
	dim3 numBlocks;
	LaunchOptions(dim3 _threadsPerBlock, dim3 _numBlocks) {
		threadsPerBlock = _threadsPerBlock; numBlocks = _numBlocks;
	}
};
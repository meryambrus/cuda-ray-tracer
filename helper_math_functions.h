#pragma once
#include <algorithm>

int getGreatestCommonFactor(int a, int b) {
	int result = 0;
	if (a < b) {
		std::swap(a, b);
	}
	for (int i = 1; i <= b; ++i) {
		if (a % i == 0 && b % i == 0) {
			result = i;
		}
	}
	return result;
}
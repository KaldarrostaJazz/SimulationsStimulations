#include "vehicle.hpp"
#include <iostream>
#include <cmath>

double F(double x, double v_max, double* par){
	return v_max*(std::tanh(par[0]*x-par[1])+std::tanh(par[1]));
}

int main() {
	double V_MAX = 0.5;
	double par[2] = {1, 1};
	double delay = 0.1;
	int N = 5;
	Vehicle vehicles[N];
}

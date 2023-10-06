#include <stdexcept>
class Vehicle {
	private:
		double x;
		double v;
	public:
		Vehicle(): x(0), v(0) {}
		Vehicle(double position, double velocity): x(position), v(velocity) {}

		double GetX();
		double GetV();

		void SetX(double pos);
		void SetV(double vel);
};

class VehicleArray {
	private:
		int N;
		Vehicle vehicle[];
};

double Vehicle::GetX() {
	return Vehicle::x;
}
double Vehicle::GetV() {
	return Vehicle::v;
}
void Vehicle::SetX(double pos) {
	Vehicle::x = pos;
}
void Vehicle::SetV(double vel) {
	Vehicle::v = vel;
}

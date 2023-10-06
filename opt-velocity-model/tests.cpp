#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "vehicle.hpp"
#include "/home/giuseppesguera/doctest.h"

TEST_CASE("TEST N. 1") {
	Vehicle A;
	Vehicle B{1.5, 0.7};
	CHECK(A.GetX() == 0); 
	CHECK(A.GetV() == 0);
	CHECK(B.GetX() == 1.5);
	CHECK(B.GetV() == 0.7);
	A.SetX(4.6);
	A.SetV(9.8);
	CHECK(A.GetX() == 4.6);
	CHECK(A.GetV() == 9.8);
}
TEST_CASE("TEST N. 2") {
}

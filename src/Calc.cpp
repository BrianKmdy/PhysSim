#include "Calc.h"

//double direction(Particle p1, Particle p2) {
//    return atan2(p2.getY() - p1.getY(), p2.getX() - p1.getX());
//}

double distance(Particle p1, Particle p2) {
    return sqrt(pow(p2.getY() - p1.getY(), 2.0f) + pow(p2.getX() - p1.getX(), 2.0f) + pow(p2.getZ() - p1.getZ(), 2.0f));
}

double distance(Vector p1, Vector p2) {
	return sqrt(pow(p2.getY() - p1.getY(), 2.0f) + pow(p2.getX() - p1.getX(), 2.0f) + pow(p2.getZ() - p1.getZ(), 2.0f));
}

int randNeg() {
    if (rand() % 2 == 0)
        return 1;
    else
        return -1;
}
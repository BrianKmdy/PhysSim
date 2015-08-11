#include "Calc.h"

bool contains(Rectangle r1, Particle p1) {
    if ((p1.getX() >= r1.getX()) &&
        (p1.getY() >= r1.getY()) &&
        (p1.getX() <= r1.getX() + r1.getWidth()) &&
        (p1.getY() <= r1.getY() + r1.getHeight()))
            return true;
    else
        return false;
}

double direction(Particle p1, Particle p2) {
    return atan2(p2.getY() - p1.getY(), p2.getX() - p1.getX());
}

double distance(Particle p1, Particle p2) {
    return sqrt(pow(p2.getY() - p1.getY(), 2.0f) + pow(p2.getX() - p1.getX(), 2.0f));
}

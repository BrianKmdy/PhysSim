#ifndef CALC_H
#define CALC_H

#include "Particle.h"
#include "Rectangle.h"

bool contains(Rectangle r1, Particle p1);
double direction(Particle p1, Particle p2);
double distance(Particle p1, Particle p2);

#endif

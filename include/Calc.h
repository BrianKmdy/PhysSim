#ifndef CALC_H
#define CALC_H

#include "Particle.h"

#define MIN_DIST 0.1
#define G 1.0
#define FORCE_CONST 0.000001

//float direction(Particle p1, Particle p2);
float distance(Particle p1, Particle p2, int step);
float distance(Vector p1, Vector p2);

int randNeg();

#endif

#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

#include "Vector.h"

class Particle
{
    private:
        double x;
        double y;
        double mass;
        Vector velocity;
        std::vector<Vector> variableForces;
        std::vector<Vector> constantForces;

        bool significant = true;

    public:
        Particle(double x = 0.0, double y = 0.0, double mass = 0.0, Vector velocity = Vector());

        void setX(double x);
        void setY(double y);
        double getX();
        double getY();
        void setMass(double mass);
        void setVelocity(Vector velocity);
        void addVariableForce(Vector force);
        void addConstantForce(Vector force);
        double getMass();
        Vector getVelocity();
        std::vector<Vector> getVariableForces();
        std::vector<Vector> getConstantForces();
        void removeVariableForces();
        void removeConstantForces();
        int getType();
        void setSignificant(bool significant);
        bool isSignificant();

        void think(double timeElapsed);
};

#endif // PARTICLE_H

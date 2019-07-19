#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

#include "Vector.h"

class Particle
{
    public:
        double x;
        double y;
        double z;
        double r;
        double g;
        double b;
        double mass;
        double radius;
        Vector velocity;
        std::vector<Vector> variableForces;
        std::vector<Vector> constantForces;

        bool significant = true;

        Particle(double x = 0.0, double y = 0.0, double z = 0.0, double mass = 0.0, Vector velocity = Vector(), double radius = 1.0);
        Particle(Vector position = Vector(), double mass = 0.0, Vector velocity = Vector(), double radius = 1.0);

        void setX(double x);
        void setY(double y);
        void setZ(double z);
        double getX();
        double getY();
        double getZ();
        void setMass(double mass);
		void setPosition(Vector position);
        void setVelocity(Vector velocity);
        void addVariableForce(Vector force);
        void addConstantForce(Vector force);
        double getMass();
        Vector getVelocity();
        double getRadius();
        void setColor(double r, double g, double b);
        double getR();
        double getG();
        double getB();
        std::vector<Vector> getVariableForces();
        std::vector<Vector> getConstantForces();
        void removeVariableForces();
        void removeConstantForces();
        int getType();
        void setSignificant(bool significant);
        bool isSignificant();

        void think(double timeElapsed, Vector force);
};

#endif // PARTICLE_H

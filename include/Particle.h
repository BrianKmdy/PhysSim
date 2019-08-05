#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

#include "Vector.h"

class Particle
{
    public:
        float r;
        float g;
        float b;
        float mass;
        float radius;
		Vector position[2];
        Vector velocity;
        std::vector<Vector> variableForces;
        std::vector<Vector> constantForces;

        bool significant = true;

        Particle(float mass = 0.0, Vector velocity = Vector(), float radius = 1.0);

        float getX(int step);
        float getY(int step);
        float getZ(int step);
        void setMass(float mass);
		void setInitialPosition(Vector position);
        void setVelocity(Vector velocity);
		void setRadius(float radius);
        void addVariableForce(Vector force);
        void addConstantForce(Vector force);
        float getMass();
        Vector getVelocity();
        float getRadius();
        void setColor(float r, float g, float b);
        float getR();
        float getG();
        float getB();
        std::vector<Vector> getVariableForces();
        std::vector<Vector> getConstantForces();
        void removeVariableForces();
        void removeConstantForces();
        int getType();
        void setSignificant(bool significant);
        bool isSignificant();

        void think(float timeElapsed, Vector force);
};

#endif // PARTICLE_H

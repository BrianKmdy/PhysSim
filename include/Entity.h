#ifndef ENTITY_H
#define ENTITY_H

#include <vector>

#include "Vector.h"

class Entity
{
    private:
        int type;

    protected:
        float mass;
        Vector velocity;
        std::vector<Vector> variableForces;
        std::vector<Vector> constantForces;

    public:
        const static int PARTICLE  = 0;
        const static int RECTANGLE = 1;
        const static int ELLIPSE  = 2;

        Entity(float mass = 0.0, Vector velocity = Vector());

        void setMass(float mass);
        void setVelocity(Vector velocity);
        void addVariableForce(Vector force);
        void addConstantForce(Vector force);
        float getMass();
        Vector getVelocity();
        std::vector<Vector> getVariableForces();
        std::vector<Vector> getConstantForces();
        void removeVariableForces();
        void removeConstantForces();
        int getType();

        virtual void think(float timeElapsed, Vector force) = 0;
};

#endif // ENTITY_H

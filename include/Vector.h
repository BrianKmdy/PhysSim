#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>

#pragma pack(push, 1)

class Vector
{
    public:
        //float magnitude;
        //float direction;
        float x;
        float y;
        float z;

        //Vector(float magnitude = 0.0, float direction = 0.0);
        Vector(float x = 0.0, float y = 0.0, float z = 0.0);

        //void setMagnitude(float magnitude);
        //void setDirection(float direction);
        void setComponents(float x, float y, float z);
        float getMagnitude();
        //float getDirection();
        float getX();
        float getY();
        float getZ();
        
        Vector normalize();

        Vector sum(Vector vector);
        Vector operator+(const Vector& vector);
        Vector operator+=(const Vector& vector);

        Vector difference(Vector vector);
        float dProduct(Vector vector);
        Vector vProduct(Vector vector);

        Vector orthogonal();
        Vector product(float scalar);
};

#pragma pack(pop)

#endif // VECTOR_H

#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>

class Vector
{
    private:
        //double magnitude;
        //double direction;
        double x;
        double y;
        double z;

    public:
        //Vector(double magnitude = 0.0, double direction = 0.0);
        Vector(double x = 0.0, double y = 0.0, double z = 0.0);

        //void setMagnitude(double magnitude);
        //void setDirection(double direction);
        void setComponents(double x, double y, double z);
        double getMagnitude();
        //double getDirection();
        double getX();
        double getY();
        double getZ();
        
        Vector normalize();

        Vector sum(Vector vector);
        Vector operator+(const Vector& vector);
        Vector operator+=(const Vector& vector);

        Vector difference(Vector vector);
        Vector sProduct(Vector vector);
        Vector vProduct(Vector vector);

        Vector orthogonal();
        Vector product(double scalar);
};

#endif // VECTOR_H

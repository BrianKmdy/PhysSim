#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>

class Vector
{
    private:
        double magnitude;
        double direction;

    public:
        Vector(double magnitude = 0.0, double direction = 0.0);

        void setMagnitude(double magnitude);
        void setDirection(double direction);
        void setComponents(double x, double y);
        double getMagnitude();
        double getDirection();
        double getX();
        double getY();

        Vector sum(Vector vector);
        Vector operator+(const Vector& vector);
        Vector operator+=(const Vector& vector);

        Vector difference(Vector vector);
        Vector sProduct(Vector vector);
        Vector vProduct(Vector vector);
};

#endif // VECTOR_H

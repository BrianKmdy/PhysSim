#ifndef RECTANGLE_H
#define RECTANGLE_H

#include "Vector.h"
#include "Particle.h"

class Rectangle : public Particle
{
    private:
        double width;
        double height;

    public:
        Rectangle(double x = 0.0, double y = 0.0, double width = 0.0, double height = 0.0, double mass = 0.0, Vector velocity = Vector());

        void setWidth(double width);
        void setHeight(double height);
        double getWidth();
        double getHeight();
};

#endif // RECTANGLE_H

#ifndef RECTANGLE_H
#define RECTANGLE_H

#include "Vector.h"
#include "Particle.h"

class Rectangle : public Particle
{
    private:
        float width;
        float height;

    public:
        Rectangle(float x = 0.0, float y = 0.0, float z = 0.0, float width = 0.0, float height = 0.0, float mass = 0.0, Vector velocity = Vector());

        void setWidth(float width);
        void setHeight(float height);
        float getWidth();
        float getHeight();
};

#endif // RECTANGLE_H

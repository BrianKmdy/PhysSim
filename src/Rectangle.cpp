#include "Rectangle.h"

Rectangle::Rectangle(float x, float y, float z, float width, float height, float mass, Vector velocity) : Particle(mass, velocity) {
    this->width = width;
    this->height = height;
}

void Rectangle::setWidth(float width) {
    this->width = width;
}

void Rectangle::setHeight(float height) {
    this->height = height;
}

float Rectangle::getWidth() {
    return width;
}

float Rectangle::getHeight() {
    return height;
}

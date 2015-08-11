#include "Rectangle.h"

Rectangle::Rectangle(double x, double y, double width, double height, double mass, Vector velocity) : Particle(x, y, mass, velocity) {
    this->width = width;
    this->height = height;
}

void Rectangle::setWidth(double width) {
    this->width = width;
}

void Rectangle::setHeight(double height) {
    this->height = height;
}

double Rectangle::getWidth() {
    return width;
}

double Rectangle::getHeight() {
    return height;
}

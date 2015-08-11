#include "Vector.h"

Vector::Vector(double magnitude, double direction) {
    this->magnitude = magnitude;
    this->direction = direction;
}

void Vector::setMagnitude(double magnitude) {
    this->magnitude = magnitude;
}

void Vector::setDirection(double direction) {
    this->direction = direction;
}

void Vector::setComponents(double x, double y) {
    magnitude = sqrt(pow(x, 2.0) + pow(y, 2.0));
    direction = atan2(y, x);
}

double Vector::getMagnitude() {
    return magnitude;
}

double Vector::getDirection() {
    return direction;
}

double Vector::getX() {
    return magnitude * cos(direction);
}

double Vector::getY() {
    return magnitude * sin(direction);
}

Vector Vector::sum(Vector vector) {
    double x1 = getX();
    double y1 = getY();
    double x2 = vector.getX();
    double y2 = vector.getY();

    return Vector(sqrt(pow(x1 + x2, 2.0) + pow(y1 + y2, 2.0)), atan2(y1 + y2, x1 + x2));
}

Vector Vector::operator+(const Vector& vector) {
    return sum(vector);
}

Vector Vector::operator+=(const Vector& vector) {
    *this = sum(vector);

    return *this;
}

Vector Vector::difference(Vector vector) {
    // TODO
}

Vector Vector::sProduct(Vector vector) {
    // TODO
}

Vector Vector::vProduct(Vector vector) {
    // TODO
}

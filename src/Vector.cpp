#include "Vector.h"

#include <cstdlib>
#include <iostream>

//Vector::Vector(double magnitude, double direction) {
//    this->magnitude = magnitude;
//    this->direction = direction;
//}

Vector::Vector(double x, double y, double z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

//void Vector::setMagnitude(double magnitude) {
//    this->magnitude = magnitude;
//}
//
//void Vector::setDirection(double direction) {
//    this->direction = direction;
//}

void Vector::setComponents(double x, double y, double z) {
    //magnitude = sqrt(pow(x, 2.0) + pow(y, 2.0));
    //direction = atan2(y, x);
    this->x = x;
    this->y = y;
    this->z = z;
}

double Vector::getMagnitude() {
    //return magnitude;
    return sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0));
}
//
//double Vector::getDirection() {
//    return direction;
//}

double Vector::getX() {
    //return magnitude * cos(direction);
    return x;
}

double Vector::getY() {
    //return magnitude * sin(direction);
    return y;
}

double Vector::getZ() {
    return z;
}

Vector Vector::normalize() {
    double magnitude = getMagnitude();

    return Vector(x / magnitude, y / magnitude, z / magnitude);
}

Vector Vector::sum(Vector vector) {
    //double x1 = getX();
    //double y1 = getY();
    double x2 = vector.getX();
    double y2 = vector.getY();
    double z2 = vector.getZ();

    return Vector(x + x2, y + y2, z + z2);
}

Vector Vector::operator+(const Vector& vector) {
    return sum(vector);
}

Vector Vector::operator+=(const Vector& vector) {
    *this = sum(vector);

    return *this;
}

Vector Vector::difference(Vector vector) {
    double x2 = vector.getX();
    double y2 = vector.getY();
    double z2 = vector.getZ();

    return Vector(x - x2, y - y2, z - z2);
}

double Vector::dProduct(Vector vector) {
	return (x * vector.getX()) + (y * vector.getY()) + (z * vector.getZ());
}

Vector Vector::vProduct(Vector vector) {
    double x2 = vector.getX();
    double y2 = vector.getY();
    double z2 = vector.getZ();

    return Vector(y * z2 - y2 * z, z * x2 - z2 * x, x * y2 - x2 * y);
}

Vector Vector::orthogonal() {
    double x2 = 0;
    double y2 = 0;
    double z2 = 0;

    if (z != 0) {
        x2 = (double) rand() / (double) RAND_MAX;
        y2 = (double) rand() / (double) RAND_MAX;
        z2 = (-x * x2 - y * y2) / z;
    } else if (y != 0) {
        x2 = (double) rand() / (double) RAND_MAX;
        z2 = (double) rand() / (double) RAND_MAX;
        y2 = (-x * x2 - z * z2) / y;
    } else if (x != 0) {
        z2 = (double) rand() / (double) RAND_MAX;
        y2 = (double) rand() / (double) RAND_MAX;
        x2 = (-z * z2 - y * y2) / x;
    } 

    Vector v = Vector(x2, y2, z2);

    return v.normalize();
}

Vector Vector::product(double scalar) {
    return Vector(scalar * x, scalar * y, scalar * z);
}
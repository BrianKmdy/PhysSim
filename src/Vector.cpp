#include "Vector.h"

#include <cstdlib>
#include <iostream>

//Vector::Vector(float magnitude, float direction) {
//    this->magnitude = magnitude;
//    this->direction = direction;
//}

Vector::Vector(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

//void Vector::setMagnitude(float magnitude) {
//    this->magnitude = magnitude;
//}
//
//void Vector::setDirection(float direction) {
//    this->direction = direction;
//}

void Vector::setComponents(float x, float y, float z) {
    //magnitude = sqrt(pow(x, 2.0) + pow(y, 2.0));
    //direction = atan2(y, x);
    this->x = x;
    this->y = y;
    this->z = z;
}

float Vector::getMagnitude() {
    //return magnitude;
    return sqrt(pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0));
}
//
//float Vector::getDirection() {
//    return direction;
//}

float Vector::getX() {
    //return magnitude * cos(direction);
    return x;
}

float Vector::getY() {
    //return magnitude * sin(direction);
    return y;
}

float Vector::getZ() {
    return z;
}

Vector Vector::normalize() {
    float magnitude = getMagnitude();

    return Vector(x / magnitude, y / magnitude, z / magnitude);
}

Vector Vector::sum(Vector vector) {
    //float x1 = getX();
    //float y1 = getY();
    float x2 = vector.getX();
    float y2 = vector.getY();
    float z2 = vector.getZ();

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
    float x2 = vector.getX();
    float y2 = vector.getY();
    float z2 = vector.getZ();

    return Vector(x - x2, y - y2, z - z2);
}

float Vector::dProduct(Vector vector) {
	return (x * vector.getX()) + (y * vector.getY()) + (z * vector.getZ());
}

Vector Vector::vProduct(Vector vector) {
    float x2 = vector.getX();
    float y2 = vector.getY();
    float z2 = vector.getZ();

    return Vector(y * z2 - y2 * z, z * x2 - z2 * x, x * y2 - x2 * y);
}

Vector Vector::orthogonal() {
    float x2 = 0;
    float y2 = 0;
    float z2 = 0;

    if (z != 0) {
        x2 = (float) rand() / (float) RAND_MAX;
        y2 = (float) rand() / (float) RAND_MAX;
        z2 = (-x * x2 - y * y2) / z;
    } else if (y != 0) {
        x2 = (float) rand() / (float) RAND_MAX;
        z2 = (float) rand() / (float) RAND_MAX;
        y2 = (-x * x2 - z * z2) / y;
    } else if (x != 0) {
        z2 = (float) rand() / (float) RAND_MAX;
        y2 = (float) rand() / (float) RAND_MAX;
        x2 = (-z * z2 - y * y2) / x;
    } 

    Vector v = Vector(x2, y2, z2);

    return v.normalize();
}

Vector Vector::product(float scalar) {
    return Vector(scalar * x, scalar * y, scalar * z);
}
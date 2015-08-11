#include "Particle.h"

Particle::Particle(double x, double y, double mass, Vector velocity)
{
    this->x = x;
    this->y = y;
    this->mass = mass;
    this->velocity = velocity;
    variableForces = std::vector<Vector>();
    constantForces = std::vector<Vector>();
}

void Particle::setMass(double mass) {
    this->mass = mass;
}

void Particle::setVelocity(Vector velocity) {
    this->velocity = velocity;
}

void Particle::addVariableForce(Vector force) {
    variableForces.push_back(force);
}

void Particle::addConstantForce(Vector force) {
    constantForces.push_back(force);
}

double Particle::getMass() {
    return mass;
}

Vector Particle::getVelocity() {
    return velocity;
}

std::vector<Vector> Particle::getVariableForces() {
    return variableForces;
}

std::vector<Vector> Particle::getConstantForces() {
    return constantForces;
}

void Particle::removeVariableForces() {
    variableForces.clear();
}

void Particle::removeConstantForces() {
    constantForces.clear();
}

void Particle::setX(double x) {
    this->x = x;
}

void Particle::setY(double y) {
    this->y = y;
}

double Particle::getX() {
    return x;
}

double Particle::getY() {
    return y;
}

void Particle::setSignificant(bool significant) {
    this->significant = significant;
}

bool Particle::isSignificant() {
    return significant;
}

void Particle::think(double timeElapsed) {
    Vector totalForce = Vector();

    for (std::vector<Vector>::iterator it = constantForces.begin(); it != constantForces.end(); ++it)
        totalForce += *it;
    for (std::vector<Vector>::iterator it = variableForces.begin(); it != variableForces.end(); ++it)
        totalForce += *it;

    Vector acceleration = Vector(totalForce.getMagnitude() / mass, totalForce.getDirection());

    x = x + (velocity.getX() * timeElapsed) + (0.5 * (acceleration.getX()) * pow(timeElapsed, 2.0));
    y = y + (velocity.getY() * timeElapsed) + (0.5 * (acceleration.getY()) * pow(timeElapsed, 2.0));

    double vX = velocity.getX() + (acceleration.getX() * timeElapsed);
    double vY = velocity.getY() + (acceleration.getY() * timeElapsed);
    velocity.setComponents(vX, vY);

    removeVariableForces();
}

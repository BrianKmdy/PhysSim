#include "Particle.h"

Particle::Particle(float mass, Vector velocity, float radius)
{
    this->mass = mass;
    this->radius = radius;
    this->velocity = velocity;
    variableForces = std::vector<Vector>();
    constantForces = std::vector<Vector>();
    r = 1;
    g = 1;
    b = 1;
}

void Particle::setMass(float mass) {
    this->mass = mass;
}

void Particle::setInitialPosition(Vector position) {
	this->position[0] = position;
	this->position[1] = position;
}

void Particle::setVelocity(Vector velocity) {
    this->velocity = velocity;
}

void Particle::setRadius(float radius) {
	this->radius = radius;
}

void Particle::addVariableForce(Vector force) {
    variableForces.push_back(force);
}

void Particle::addConstantForce(Vector force) {
    constantForces.push_back(force);
}

float Particle::getMass() {
    return mass;
}

float Particle::getRadius() { 
    return radius;
}

Vector Particle::getVelocity() {
    return velocity;
}
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

void Particle::setColor(float r, float g, float b) {
    this->r = r;
    this->g = g;
    this->b = b;
}

float Particle::getR() {
    return r;
}

float Particle::getG() {
    return g;
}

float Particle::getB() {
    return b;
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

float Particle::getX(int step) {
    return position[step % 2].x;
}

float Particle::getY(int step) {
	return position[step % 2].y;
}

float Particle::getZ(int step) {
	return position[step % 2].z;
}

void Particle::setSignificant(bool significant) {
    this->significant = significant;
}

bool Particle::isSignificant() {
    return significant;
}

// void Particle::think(float timeElapsed, Vector force) {
//     //Vector acceleration = Vector(totalForce.getMagnitude() / mass, totalForce.getDirection());
//     Vector acceleration = Vector(force.getX() / mass, force.getY() / mass, force.getZ() / mass);
// 
//     x = x + (velocity.getX() * timeElapsed) + (0.5 * (acceleration.getX()) * pow(timeElapsed, 2.0));
//     y = y + (velocity.getY() * timeElapsed) + (0.5 * (acceleration.getY()) * pow(timeElapsed, 2.0));
//     z = z + (velocity.getZ() * timeElapsed) + (0.5 * (acceleration.getZ()) * pow(timeElapsed, 2.0));
// 
//     float vX = velocity.getX() + (acceleration.getX() * timeElapsed);
//     float vY = velocity.getY() + (acceleration.getY() * timeElapsed);
//     float vZ = velocity.getZ() + (acceleration.getZ() * timeElapsed);
//     velocity.setComponents(vX, vY, vZ);
// }

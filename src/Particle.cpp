#include "Particle.h"

Particle::Particle(double x, double y, double z, double mass, Vector velocity, double radius)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->mass = mass;
    this->radius = radius;
    this->velocity = velocity;
    variableForces = std::vector<Vector>();
    constantForces = std::vector<Vector>();
    r = 1;
    g = 1;
    b = 1;
}

Particle::Particle(Vector position, double mass, Vector velocity, double radius)
{
    this->x = position.getX();
    this->y = position.getY();
    this->z = position.getZ();
    this->mass = mass;
    this->radius = radius;
    this->velocity = velocity;
    variableForces = std::vector<Vector>();
    constantForces = std::vector<Vector>();
    r = 1;
    g = 1;
    b = 1;
}

void Particle::setMass(double mass) {
    this->mass = mass;
}

void Particle::setPosition(Vector position) {
	x = position.getX();
	y = position.getY();
	z = position.getZ();
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

double Particle::getRadius() { 
    return radius;
}

Vector Particle::getVelocity() {
    return velocity;
}

void Particle::setColor(double r, double g, double b) {
    this->r = r;
    this->g = g;
    this->b = b;
}

double Particle::getR() {
    return r;
}

double Particle::getG() {
    return g;
}

double Particle::getB() {
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

void Particle::setX(double x) {
    this->x = x;
}

void Particle::setY(double y) {
    this->y = y;
}

void Particle::setZ(double z) {
    this->z = z;
}

double Particle::getX() {
    return x;
}

double Particle::getY() {
    return y;
}

double Particle::getZ() {
    return z;
}

void Particle::setSignificant(bool significant) {
    this->significant = significant;
}

bool Particle::isSignificant() {
    return significant;
}

void Particle::think(double timeElapsed, Vector force) {
    //Vector acceleration = Vector(totalForce.getMagnitude() / mass, totalForce.getDirection());
    Vector acceleration = Vector(force.getX() / mass, force.getY() / mass, force.getZ() / mass);

    x = x + (velocity.getX() * timeElapsed) + (0.5 * (acceleration.getX()) * pow(timeElapsed, 2.0));
    y = y + (velocity.getY() * timeElapsed) + (0.5 * (acceleration.getY()) * pow(timeElapsed, 2.0));
    z = z + (velocity.getZ() * timeElapsed) + (0.5 * (acceleration.getZ()) * pow(timeElapsed, 2.0));

    double vX = velocity.getX() + (acceleration.getX() * timeElapsed);
    double vY = velocity.getY() + (acceleration.getY() * timeElapsed);
    double vZ = velocity.getZ() + (acceleration.getZ() * timeElapsed);
    velocity.setComponents(vX, vY, vZ);
}

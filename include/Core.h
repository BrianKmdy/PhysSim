#ifndef CORE_H
#define CORE_H

// I propose a new fundamental force as the possible source for gravity. I imagine that everywhere in space there is a fluid
// called the cosmic medium which is infinitely compressible and seeks to spread out as much as possible. What we recognize
// as gravity could simply be the absence of this force between bodies of mass. Note that this hypothesis is inspired by
// ideas such as dark energy and integrates the concept of an expanding universe
//
// Hypothesis of the universal medium
// 1. At the time of the big bang the cosmic medium was an infinitely compressed fluid
// 2. Between any infinitesimal division of the fluid there is a repulsive force causing it to spread apart
// 3. The fluid spread apart rapidly leading to areas of uniform density in the center of the universe and aread of
//    high density at the outer edges of the expanding universe
// 4. Over time as the density becomes uniform the fluid begins to condense into matter
// 5. Since matter is highly concentrated relative to the fluid its repulsive force is also much stronger
// 6. The highly condensed matter creates small bubbles or pockets around it with lower density fluid
// 7. Rather than gravity being an attractive force between matter it's actually just the lack of a repulsive force
//    as these pockets create low density areas of fluid between them
// 8. As the low density pockets of fluid form the matter is squeezed together by the fluid surrounding it
// 9. The fluid is also the medium through which electromagnic waves travel at the speed of light
// 10. Action at a distance is impossible (classical gravity), any force which acts over distance must travel through the medium
//
// Further testing needs to be done to prove the validity of this hypothesis
// ~Brian Kimball Moody
// 11/14/2019

#include <vector>

#include "Simulate.cuh"
#include "Types.cuh"

class Core
{
public:
	Core();
    Core(Instance* instance);
	~Core();

	Instance* getInstance();

	void run();
private:
	Instance* instance;
};

#endif // CORE_H

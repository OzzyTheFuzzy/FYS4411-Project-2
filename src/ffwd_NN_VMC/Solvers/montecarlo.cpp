#include "montecarlo.h"
#include "Math/random.h"


MonteCarlo::MonteCarlo(std::unique_ptr<class Random> rng, bool preferAnalytic, bool useCache)
    : m_rng(std::move(rng)), m_preferAnalytic(preferAnalytic), m_useCache(useCache) {}

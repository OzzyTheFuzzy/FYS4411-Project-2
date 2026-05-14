## Compiling the C++ project
In order to compile the project, you can use the updated bash scripts
by running "sh ./build_debug.sh" or "sh ./build_release.sh".
 - ./build_debug.sh   : activates the following flag: "-g"
                        and generates the executable ./vmc_debug
 - ./build_release.sh : activates the following flags:
                        "-O3 -march=native -flto -fno-math-errno"
                        and generates the executable ./vmc_release

## Where is the relevant output found?
A concise summary of the output is printed to terminal.
A detailed log file (saved with time and date) is generated at each run in folder ./logs .
Other specific output is explained below.

## Running the executable
The program executes in 3 steps (all ):

    1 - Optimization  : starting from an initial guess for the parameters 
                        ( initialParams in main() ), executes the BFGS optimization; 
                        ./iofiles/log.csv is updated (and flushed) at each parameter guess
                        so you can follow the process, since it can take a while;
                        the final parameters are also printed to ./iofiles/params.dat .

    2 - FinalMC &     : runs a MC simulation sampling energies, setting up the
        blocking        wavefunction with the parameters found in ./iofiles/params.dat ;
                        the local energy value calculated at each step is printed to
                        ./iofiles/finalMCenergies.dat ;
                        after that, the blocking algorithm runs reading from
                        ./iofiles/finalMCenergies.dat and printing results also
                        to ./iofiles/blocking_results.csv

    3 - One-body      : runs a MC simulation sampling the position of the particles
        density         to build a histogram;
                        the results (the one-body density and the Poisson esitmate
                        of its error, the radial probability and the Poisson estimate 
                        of its error) are printed to ./iofiles/onebodydensity.dat

If you run the executable with no extra arguments (e.g., "./vmc_release"),
all 3 steps will run one after the other.
You can also choose to run only one or two of the steps, specifying their corresponding
numbers as extra arguments to the command line:
    e.g. #1: if you want to run only step 2, you can run "./vmc_release 2";
    e.g. #2: if you want to run only steps 1 and 3, you can run "./vmc_release 1 3" 
             or, equivalently, "./vmc_release 3 1".

## Choosing the system and the parameters
Although quite long, the main() structure should be easily readable and editable at will.
You are probably only interested in the first section called "--- Parameters ---".
There you can tune all parameters of major interest.
I hope the variables' names are comprehensible enough.
Here is a rundown of the trickiest ones:
 -  The first three strings select the Hamiltonian subclass to use, the WaveFunction subclass to use
    and the MonteCarlo solver to use. The comments on the side list the currently available subclasses.
    MonteCarlo's subclass named Metropolis refers to brute-force sampling.
    Note that if you want to add and use a newly implemented subclass, you need to update
    the 'if-cluster' in the "--- Setup Factories---" below.
 -  preferAnalytic is a boolean flag assigned to the solver, which instructs the hamiltonian
    to either ask for an analytical derivative to the wavefunction class or resort forcibly
    to the numerical implementations provided by default.
    Note that the way this is implemented requires every WaveFunction's subclass to override
    the virtual method " bool hasAnalyticalDerivative() ".
 -  repulsive_a_factor sets the hard-core diamter a according to
        a = repulsive_a_factor / sqrt(omega)
 -  seed is the seed used for the random number generator and works as follows:
    if seed == 0, then every MonteCarlo run extracts a new random seed using
        std::chrono::system_clock::now().time_since_epoch().count();
    else every MonteCarlo run runs on the same fixed seed.
    You can experience a better chance of convergence in the BFGS optimization if 
    you choose a fixed seed, maybe declaring%initializing seed in main() using
        int seed = chrono::system_clock::now().time_since_epoch().count();
    so that every program run (not MC run!) uses a different seed.
 -  While initialParams usage is straight forward, I recommend providing only resonable
    initial guesses, otherwise criticalities may easily arise (I think mainly due to numerical
    derivatives' instabilities). To prevent this from happening during the BFGS optimization
    process, wavefunction classes implement two constant methods (e.g.)
        std::vector<double> lowerBounds() const override { return { 1e-3, 0.1 }; }
        std::vector<double> upperBounds() const override { return { 1.0, 5.0 }; }
    which work as bounds for the optimization process.
    Note that, referring to the project report, for the simple harmonic oscillator results
    I set the initial alpha to 0.75, for the non-interacting elliptical case I set
    (alpha, beta) = { 0.55, 3.00 }, and for the interacting elliptical case I set
    (alpha, beta) = { 0.50, 2.8243 }.
 -  In VMCOptimizer.cpp, there are two hard-codings:
        lib_optimizer.set_maxeval(400);
        lib_optimizer.set_maxtime(10800.0);
    these set, respectively, the maximum number of parameter guesses and
    the maximum run time conceded to NLopt's optimization.
    Removing their hard-coding isn't much work, but I didn't think it would be worth the effort.
    I'm just stating this for the sake of completeness and clarity.

## Jupyter Notebook
A Jupyter Notebook display.ipynb is provided to analyze the final results visually.
It is not well-refined - it was only a handy tool.
The first cells are the most reusable ones, whereas the last ones were used to finalize
the plots for the project report and read from old folders.
The first few cells read .iofiles/log.csv in order to plot the optimization
evolution in 2D (for 1 variational parameters) and in 3D (for 2 variational parameters).
You can adjust the burnin steps to your own liking.
The other cells read .iofiles/onebodydensity.dat in order to plot
the one-body density and radial probability plots.

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
If a neural network is optimized, a file containing the optimization steps is output
(saved with time and date) in folder ./logs_NN .
If One-body density is run, a file is output (saved with time and date) in folder ./logs_OBD .
Other specific output is explained below.

## Running the executable
The program executes in 3 steps (all ):

    1 - Optimization  : if no neural network is used:
                            starting from an initial guess for the parameters 
                            ( initialParams in main() ), executes the BFGS optimization; 
                            ./iofiles/log.csv is updated (and flushed) at each parameter guess
                            so you can follow the process, since it can take a while;
                            the final parameters are also printed to ./iofiles/params.dat .
                        if a neural network is used as waveFunctionType:
                            the neural network is pre-trained to behave like the wavefunction
                            chosen as waveFunctionTrainType, then it is trained to minimize
                            the energy; the file generated in ./logs_NN is updated
                            (and flushed) at each parameter guess so you can follow the
                            process, since it can take a while; the final parameters
                            are also printed to ./iofiles/params.dat .

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
                        of its error) are printed to the file generated in ./logs_OBD

If you run the executable with no extra arguments (e.g., "./vmc_release"),
all 3 steps will run one after the other.
You can also choose to run only one or two of the steps, specifying their corresponding
numbers as extra arguments to the command line:
    e.g. #1: if you want to run only step 2, you can run "./vmc_release 2";
    e.g. #2: if you want to run only steps 1 and 3, you can run "./vmc_release 1 3" 
             or, equivalently, "./vmc_release 3 1".

## Choosing the system and the parameters
A sample config.json file is provided: it comprises and sets all the tunable parameters of the
program. Here follows a quick rundown of their usage, and their meaning for the trickiest ones.
 # System parameters
 -  hamiltonianType:
        accepted values:    "HarmonicOscillator" or "RepulsiveHO" or "CoulombHO".
        description:        RepulsiveHO is a harmonic oscillator with an interaction
                            hardcore diameter set by the physical parameters repulsive_a_factor
                            and repulsive_strength. CoulombHO is a harmonic oscillator
                            with a Coulomb-like interaction set by the physical parameter
                            maxStrength. Only CoulombHO is fully compatible with the neural
                            network optimization process.
 -  waveFunctionTrainType:
        accepted values:    "SimpleGaussian" or "EllipticGaussian" (or "RepEllipticGaussian")
        description:        It's the wavefunction which a neural network is pre-trained to
                            resemble. Repulsive options are discouraged.
 -  waveFunctionType:
        accepted values:    "SimpleGaussian" or "EllipticGaussian" or "RepEllipticGaussian" or 
                            "NN_envelope"
        description:        It's the system's wavefunction. If NN_envelope is chosen, the
                            optimization step follows the neural network optimization procedure.
 -  solverType:
        accepted values:    "Metropolis" or "MetropolisHastings"
        description:        It sets the kind of Monte Carlo. Metropolis selects brute force Metropolis
                            Monte Carlo. MetropolisHastings selects importance sampling 
                            MetropolisHastings Monte Carlo. This parameter conditions the 
                            interpretation of the timeStep parameter.
 -  preferAnalytic:
        accepted values:    true or false
        description:        flag assigned to the solver, which instructs the hamiltonian
                            to either ask for an analytical derivative to the wavefunction class or
                            resort forcibly to the numerical implementations provided by default.
                            Note that the way this is implemented requires every WaveFunction's
                            subclass to override the virtual method " bool hasAnalyticalDerivative() ".
 -  useCache:
        accepted values:    true or false
        description:        flag to enable wavefunction caching for importance sampling. Without
                            neural networks it might improve performances. With neural networks, it
                            must be turned to false.
 # Physics parameters
 -  dimensions:
        accepted values:    positive integers
        description:        number of dimensions
 -  particles:
        accepted values:    positive integers
        description:        number of particles
 -  omega:
        accepted values:    positive floating point
        description:        default frequency of the harmonic oscillator potentials along every axis
 -  omega_z:
        accepted values:    positive floating point
        description:        frequency of the elliptic harmonic oscillator potentials along the third axis
 -  repulsive_a_factor:
        accepted values:    positive floating point
        description:        hardcore radius for RepulsiveHO according to
                            a = repulsive_a_factor / sqrt(omega)
 -  repulsive_strength:
        accepted values:    floating point or "inf" 
        description:        value of the square well/step potential within the hardcore diameter
                            for RepulsiveHO
 -  maxStrength:
        accepted values:    floating point
        description:        scaling factor for CoulombHO (leave 1 for 1/r Coulomb repulsion)
 -  initialParams:
        accepted values:    array of floating points
        description:        initial parameters guesses for the optimization processes of
                            non-neural-network wavefunctions. The initial parameters are chosen at
                            random as explained later in this readme file.
 # Monte Carlo parameters
 -  timeStep:
        accepted values:    floating point
        description:        time step parameter for MatropolisHasting; step length parameter
                            for Metropolis
 -  equilibrationSteps:
        accepted values:    positive integer
        description:        number of equilibration steps
 -  metropolisSteps:
        accepted values:    positive integer
        description:        number of Monte Carlo steps to perform for optimization procedures
 -  finalMClog2steps
        accepted values:    positive integer
        description:        logarithm in base 2 of the number of Monte Carlo steps
                            to perform for the final run procedure
 -  BFGS_tol:
        accepted values:    positive floating point
        description:        read project 1
 # Neural Network parameters
 -  Nhid:
        accepted values:    positive integer
        description:        number of hidden nodes
 -  activationFunctionType:
        accepted values:    "tanh" or "gelu" or "sigmoid"
        description:        activation function
 -  helpDecay:
        accepted values:    floating points (range (0, 0.5) is recommended)
        description:        gaussian envelope factor (read project report)
 -  lr:
        accepted values:    positive floating point
        description:        initial learning rate for the Adam Optimizer
 -  Adam_ktol:
        accepted values:    floating point in range (0, 1]
        description:        overlap integral threshold to skip the remaining pre-training
                            steps and proceed directly to training
 -  max_patience:
        accepted values:    positive integer
        description:        Number of maximum contiguous pre-training/training
                            non-adiabatic steps without discovering any improvement
                            greater than min_improvement. If this number is reached,
                            learning rate and min_improvement are divided by 10.
 -  min_improvement:
        accepted values:    positive floating point
        description:        Minimum improvement to discover within a max_patience number of
                            contiguous pre-training/training non-adiabatic steps, before
                            dividing learning rate and min_improvement by 10.
 -  nPretrainSteps:
        accepted values:    positive integer
        description:        number of pre-training steps
 -  nEnergySteps:
        accepted values:    positive integer
        description:        number of training steps
 -  nAdiabSteps:
        accepted values:    integer less than nEnergySteps
        description:        number of training steps over which linearly and adiabatically
                            increase the interaction potential's strength from 0%
                            to 100% (wrt its maxStrength)
 # Observables parameters
 -  onebodyDensitySteps:
        accepted values:    positive integer
        description:        number of Monte Carlo steps to perform for the One-body density procedure
 -  onebodyDensity_rMax:
        accepted values:    positive floating point
        description:        maximum distance from the origin where to set the last bin of the
                            One-body density histogram 
 -  onebodyDensity_nBins:
        accepted values:    positive integer
        description:        number of bins for the One-body density histogram
 # Miscellaneous parameters
 -  seed:
        accepted values:    integer
        description:        if seed == 0, then every MonteCarlo run extracts a new random seed using
                            std::chrono::system_clock::now().time_since_epoch().count();
                            else every MonteCarlo run runs on the same fixed seed.

 -  In VMCOptimizer.cpp, there are two hard-codings:
        lib_optimizer.set_maxeval(400);
        lib_optimizer.set_maxtime(10800.0);
    these set, respectively, the maximum number of parameter guesses and
    the maximum run time conceded to NLopt's optimization.
    Removing their hard-coding isn't much work, but I didn't think it would be worth the effort.
    I'm just stating this for the sake of completeness and clarity.
 -  In neuralnetwork.cpp, there is a hard-coding:
    if no parameters are passed to the constructor of a NeuralNetwork,
    all weights are initialized at random, according to:
        double scale1 = std::sqrt(2.0 / (Nin + Nhid)) * 0.01;
        double scale2 = std::sqrt(2.0 / Nhid) * 0.01;
        scale2 = scale1 = 1;
        m_W1 = register_parameter("W1", torch::randn({ Nin, Nhid }, opts) * scale1);
        m_b = register_parameter("b", torch::randn(Nhid, opts) * scale2);
        m_W2 = register_parameter("W2", torch::randn(Nhid, opts) * scale2);
    This is what happens by default when a neural network is asked to be optimized.

## Jupyter Notebook (the old one)
A Jupyter Notebook display.ipynb is provided to analyze the final results visually.
It is not well-refined - it was only a handy tool.
The first cells are the most reusable ones, whereas the last ones were used to finalize
the plots for the project report and read from old folders.
The first few cells read .iofiles/log.csv in order to plot the optimization
evolution in 2D (for 1 variational parameters) and in 3D (for 2 variational parameters).
You can adjust the burnin steps to your own liking.

## Jupyter Notebook (the new one)
A Jupyter Notebook display2.ipynb is provided to analyze the final results visually.
It is not too well-refined - it was mainly a handy tool.
Nonetheless its use is more strongly recommended than display.ipynb in project 1.
The first cells are the most important ones, whereas the last ones were used to
finalize the plots for the project report and read specific files which are not
included in the git repository.
The first big cell reads the files in ./logs_NN, calculates and prints the mean
over the last 5% of the training steps (it is also found in the legend of the plots)
and saves the plots to ./figs_NN .
The second big cell reads the files in ./logs_OBD and saves the plots to ./figs_NN .
Note that the generated png's store information read from the general log file in ./logs .
corresponding to the respective run according to the filenaming system based on date
and time; this is a handy tool to compare different results quickly, but please pay
attention to the relevance of certain variables to the specific run execution.
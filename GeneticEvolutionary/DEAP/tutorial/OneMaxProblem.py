import array
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# -------------------------------------
# Creator
# -------------------------------------

'''
'creator' is a class-factory. It's first argument is the name of the new class.
It's second is the base class it will inherit. Any subsequent arguments will
become attributes of the class. In the code below, we define the class 'FitnessMax',
which inherits the Fitness class of the deap.base module and contain an additional
attribute called 'weights'. (Weights gives the number of objectives during evolution?)
'''
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

'''
Next we create the class 'Individual', which inherits the class 'list' and contains
our previously defined FitnessMax class in its fitness attribute. 
'''
creator.create("Individual", list, fitness=creator.FitnessMax)


# -------------------------------------
# Toolbox
# -------------------------------------

'''
We now use our custom classes to create types representing our individuals as well
as our whole population. All the objects we will use (i.e.. an individual, the
population, as well as all functions, operators, and arguments) will be stored in
a DEAP container called 'Toolbox'. It contains two methods for adding and removing
content: register() and unregister().
'''
toolbox = base.Toolbox()

# Attribute generator
'''Here we create a function to randomly generate an attribute (of either 0 or 1)'''
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
'''
Our individuals will be generated using the function initRepeat(). Its first argument
is a container class (here, the 'Individual' one we defined in the previous section).
The next argument is the function to be used to fill this population (here, attr_bool()
defined above). The final argument here is the number of times this function is called
for an individual). Finally, a 'population' is creating using a similar process. Here,
a list is made using the contents of "individual.
'''
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# -------------------------------------
# The evaluation function
# -------------------------------------
'''
The evaluation function is pretty simple in our example. We just need to count the number
of ones in an individual. Note that the returned value must be an iterable of a length equal
to the number of objectives (weights).
'''
def evalOneMax(individual):
    return sum(individual),


# -------------------------------------
# The genetic operators
# -------------------------------------

# register the goal/fitness function
toolbox.register("evaluate", evalOneMax)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)


# -------------------------------------
# Evolving the population
# -------------------------------------

def long():
    random.seed(64)

    # Initialise population (using 'population' function defined above)
    pop = toolbox.population(n=300)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses): # Set fitness of each individual
        ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(pop))

    # Set evolution constants
    ''' CXPB = the probability with which two individuals are crossed
        MUTPB = the probability for mutating an individual '''
    CXPB, MUTPB = 0.5, 0.2

    # -------------------------------------
    # Performing the evolution
    # -------------------------------------
    print("Start of evolution")
    fits = [ind.fitness.values[0] for ind in pop] # Get initial fitnesses of our population
    g = 0 # Variable keeping track of the number of generations

    # Begin the evolution
    while max(fits) < 100 and g < 1000: # While maximum fitness is < 100, and generations is < 1000
        
        # -------------------------------------
        # Make a new generation
        # -------------------------------------

        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        '''The toolbox.clone() method ensure that we don’t use a reference to the individuals but
        an completely independent instance. This is of utter importance since the genetic
        operators in toolbox will modify the provided objects in-place'''
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover on the offspring
        '''Note that [::2] slices every other index from 0 to end, while [1::2]
        slices every other index from 1 to end'''
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values # fitness values of the children must be recalculated later
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness (i.e. fitness has been removed above?)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind) # Evaluate fitness
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

def short():
    random.seed(64)
    
    # Create population
    pop = toolbox.population(n=300)

    # Initialise best-individual-ever store
    hof = tools.HallOfFame(maxsize=1)

    # Get statistics (about fitness)
    '''This lambda is a function to access the values on which to compute the statistics'''
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    # Run evolution!
    pop, log = algorithms.eaSimple(
        population = pop,
        toolbox = toolbox,
        cxpb = 0.5,
        mutpb = 0.2,
        ngen=40, 
        stats = stats,
        halloffame = hof,
        verbose = True
    )
    
    # return pop, log, hof

if __name__ == "__main__":
    short()
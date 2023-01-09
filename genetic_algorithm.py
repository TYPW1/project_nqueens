from collections import namedtuple
from nqueens import NQUEENS
from copy import deepcopy
import random
import numpy
import tools
import tableprint
import pandas
import click
import os

class Individual:
    """
    Class container to recorder solution and solution_value
    """
    def __init__(self,solution, fitness):
        self.solution = solution
        self.fitness = fitness

class GA:


    def __init__(self,size):
       self.problem = NQUEENS(size)


    def init_population(self,n=300):
        population = []
        for _ in range(n):
            permutation = list(range(self.problem.size))
            random.shuffle(permutation)
            population.append(Individual(solution=permutation,fitness=None))
        return population


    def evaluate(self,individual):
        """
        Return the number of conflicts on the board
        """
        return self.problem.set_nqueens(individual.solution)

    def selection(self,individuals,k,tournsize=2):
        chosen = []
        """
        Select the best individuals among the population.
        """
        chosen = []
        for _ in range(k):
            # Select candidates by tournament
            candidates = random.sample(individuals, tournsize)
            # Select the fittest candidate
            fittest = max(candidates, key=lambda ind: ind.fitness)
            # Append the fittest candidate to the list of chosen individuals
            chosen.append(fittest)
        # Return the list of tournament winners

	#
	#  Exercice 2. a) 
	#
	#  Return a list of selected candidates
	#
        return chosen

    def crossover(self,ind1,ind2):
   	#
	#  Exercice 2. b)
	#
	#  Define HERE the crossover operation
	#  
	#
        """
        Perform a single point crossover on the input individuals.
        The point of crossover is selected at random and the resulting
        individuals are returned.
        """
        # Select a random point for the crossover
        point = random.randrange(len(ind1.solution))

        # Create the offspring by combining the solutions of the parents
        offspring1 = ind1.solution[:point] + ind2.solution[point:]
        offspring2 = ind2.solution[:point] + ind1.solution[point:]

        return Individual(solution=offspring1, fitness=None), Individual(solution=offspring2, fitness=None)

    def mutation(self,individual,indpb):
        """
        Mutate an individual by randomly selecting two positions in the solution and swapping them.
        """
        size = len(individual.solution)
        for i in range(size):
            if random.random() < indpb:
                swap_index = random.randint(0, size-1)
                individual.solution[i], individual.solution[swap_index] = individual.solution[swap_index], individual.solution[i]

	#
	#  Exercice 2. c)
	#
	#  Define HERE the mutation operation
	#
        return individual



    def solve(self,npop,ngen, cxpb,mutpb,indpb,verbose=True):

        stats = tools.Statistics(lambda ind: ind.fitness)
        stats.register("Avg", numpy.mean)
        stats.register("Std", numpy.std)
        stats.register("Min", numpy.min)
        stats.register("Max", numpy.max)

        header = ['gen', 'nevals'] + (stats.fields if stats else [])
        all_gen = [header]

        # Generate initial population
        population = self.init_population(npop)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness is not None]
        fitnesses = map(self.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness = fit

        record = stats.compile(population) if stats else {}
        all_gen.append([0,len(invalid_ind)]+record)
        if verbose:
            tableprint.banner(f"NQUEENS -- Genetic algorithm solver -- size {self.problem.size}")
            print(tableprint.header(header,width=20))
            print(tableprint.row([0,len(invalid_ind)]+record,width=20), flush=True)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            next_population = self.selection(population, len(population))

            # Apply crossover and mutation on the offspring
            for i in range(1, len(next_population), 2):
                if random.random() < cxpb:
                    next_population[i - 1], next_population[i] = self.crossover(deepcopy(next_population[i - 1]),deepcopy(next_population[i]))
                    # We need now to recompute the fitness
                    next_population[i-1].fitness = None
                    next_population[i].fitness = None

            for i in range(len(next_population)):
                if random.random() < mutpb:
                    next_population[i] = self.mutation(deepcopy(next_population[i]),indpb)
                    # We need now to recompute the fitness
                    next_population[i].fitness = None

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in next_population if not ind.fitness is not None]
            fitnesses = map(self.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness = fit


            # Replace the current population by the offspring
            population[:] = next_population

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            all_gen.append([ngen+1,len(invalid_ind)]+record)
            if verbose:
                print(tableprint.row([gen+1,len(invalid_ind)]+record,width=20), flush=True)

        return population, all_gen


def validate_proba(ctx, param, value):
    if  value >= 0 and value <= 1.0:
        return value
    else:
        raise click.BadParameter(f"Wrong {param} wrong ==> 0<= {param} <=1 ")

def check_path(ctx, param, value):
    if value is None:
        return value
    if os.path.exists(value):
        raise click.BadParameter(f"Path {param} exits already; Not overriding ")
    os.mkdir(value)
    return value


@click.command()
@click.option('--size', default=10, help='Size of the nqueens problem')
@click.option('--npop', default=100, help='Number of individual in the population')
@click.option('--ngen', default=100, help='Number of generations')
@click.option('--cxpb', default=0.5, help='Crossover probability',callback=validate_proba)
@click.option('--mutpb', default=0.2,  help='Mutation probability',callback=validate_proba)
@click.option('--indpb', default=0.2,  help='Allele mutation probability',callback=validate_proba)
@click.option('--verbose/--no-verbose')
@click.option('--save', default=None,  help='Record population and generation in a non-existing directory',callback=check_path)
def main(size,npop,ngen,cxpb,mutpb,indpb,verbose,save):
    solver_nqueens = GA(size)
    last_pop,all_gen = solver_nqueens.solve(npop,ngen, cxpb,mutpb,indpb,verbose=verbose)
    if save is not None:
        data=[]
        for ind in last_pop:
            data.append(ind.solution + [ind.fitness])
        df_pop = pandas.DataFrame(data=data, columns=[f"allele {i}" for i in range(size)]+["fitness"])
        df_pop.to_excel(os.path.join(save,"population.xlsx"))
        df_log = pandas.DataFrame(data=all_gen)
        df_log.to_excel(os.path.join(save,"log.xlsx"))





if __name__ == "__main__":
    main()

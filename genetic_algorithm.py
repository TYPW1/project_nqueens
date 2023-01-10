

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
import matplotlib.pyplot as plt


class Individual:
    """
    Class container to recorder solution and solution_value
    """

    def __init__(self, solution, fitness):
        self.solution = solution
        self.fitness = fitness


class GA:

    def __init__(self, size):
        self.problem = NQUEENS(size)

    def init_population(self, n=300):
        population = []
        for _ in range(n):
            permutation = list(range(self.problem.size))
            random.shuffle(permutation)
            population.append(Individual(solution=permutation, fitness=None))
        return population

    def evaluate(self, individual):
        """
        Return the number of conflicts on the board
        """
        return self.problem.set_nqueens(individual.solution)

    def selection(self, individuals, k, tournsize=2):
        #
        #  Exercice 2. a)
        #
        #  Return a list of selected candidates
        #
        #Select the best individuals among the population.
        chosen = []

        for _ in range(k):
            # Select candidates by tournament
            rest = random.sample(individuals, tournsize)
            # Select the fittest candidate
            min_fitness = min([element.fitness for element in rest])
            selected_individuals = [
                element for element in rest if element.fitness == min_fitness]
            selected = selected_individuals
            if len(selected_individuals) > 1:
                selected = random.sample(selected_individuals, 1)
            # Append the fittest candidate to the list of chosen individuals
            chosen.append(selected[0])
        # Return the list of tournament winners
        return chosen

    def crossover(self, ind1, ind2):
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
        index = len(ind1.solution) - 1
        for i in range(len(ind1.solution)):
            # Create the offspring by combining the solutions of the parents
            if all(item not in ind2.solution[i:] for item in ind1.solution[:i + 1]):
                index = i
        tmp = ind1.solution[index:].copy()
        ind1.solution[index:] = ind2.solution[index:]
        ind2.solution[index:] = tmp
        return Individual(solution=ind1.solution, fitness=ind1.fitness), Individual(solution=ind2.solution, fitness=ind2.fitness)
  
    def mutation(self, individual, indpb):
        #
        #  Exercice 2. c)
        #
        #  Define HERE the mutation operation
        #
        # Mutate an individual by randomly selecting two positions in the solution and swapping them.
        for i in range(len(individual.solution)):
            r = random.random()
            if r <= indpb:
                individual.solution[i] = random.randint(
                    0, self.problem.size - 1)
        return individual

    def solve(self, npop, ngen, cxpb, mutpb, indpb, verbose=True):

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
        invalid_ind = [
            ind for ind in population if not ind.fitness is not None]
        fitnesses = map(self.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness = fit

        record = stats.compile(population) if stats else {}
        all_gen.append([0, len(invalid_ind)]+record)
        if verbose:
            tableprint.banner(
                f"NQUEENS -- Genetic algorithm solver -- size {self.problem.size}")
            print(tableprint.header(header, width=20))
            print(tableprint.row([0, len(invalid_ind)] +
                  record, width=20), flush=True)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            next_population = self.selection(population, len(population))

            # Apply crossover and mutation on the offspring
            for i in range(1, len(next_population), 2):
                if random.random() < cxpb:
                    next_population[i - 1], next_population[i] = self.crossover(
                        deepcopy(next_population[i - 1]), deepcopy(next_population[i]))
                    # We need now to recompute the fitness
                    next_population[i-1].fitness = None
                    next_population[i].fitness = None

            for i in range(len(next_population)):
                if random.random() < mutpb:
                    next_population[i] = self.mutation(
                        deepcopy(next_population[i]), indpb)
                    # We need now to recompute the fitness
                    next_population[i].fitness = None

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [
                ind for ind in next_population if not ind.fitness is not None]
            fitnesses = map(self.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness = fit
                # Replace the current population by the offspring
            population[:] = next_population

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            all_gen.append([ngen+1, len(invalid_ind)]+record)
            if verbose:
                print(tableprint.row(
                    [gen+1, len(invalid_ind)]+record, width=20), flush=True)

        return population, all_gen


def validate_proba(ctx, param, value):
    if value >= 0 and value <= 1.0:
        return value
    else:
        raise click.BadParameter(f"Wrong {param} wrong ==> 0<= {param} <=1 ")


def check_path(ctx, param, value):
    if value is None:
        return value
    if os.path.exists(value):
        raise click.BadParameter(
            f"Path {param} exits already; Not overriding ")
    os.mkdir(value)
    return value


@click.command()
@click.option('--size', default=50, help='Size of the nqueens problem')
@click.option('--npop', default=10, help='Number of individual in the population')
@click.option('--ngen', default=100, help='Number of generations')
@click.option('--cxpb', default=0.2, help='Crossover probability', callback=validate_proba)
@click.option('--mutpb', default=0.8,  help='Mutation probability', callback=validate_proba)
@click.option('--indpb', default=0.02,  help='Allele mutation probability', callback=validate_proba)
@click.option('--verbose/--no-verbose')
@click.option('--save', default=None,  help='Record population and generation in a non-existing directory', callback=check_path)
@click.option('--save_data/--no-save_data', help='Save tables and plots')
@click.option('--print_data/--no-print_data', default=True, help='Output table and plot on STDOUT')
def main(size, npop, ngen, cxpb, mutpb, indpb, verbose, save_data, print_data, save=None):
    stats = {item: numpy.zeros(ngen) for item in {"Avg", "Std", "Max", "Min"}}
    hit_stats = {"Avg": 0, "Std": 0, "Max": 0, "Min": 0}
    max_iterations = 30
    all_gen = list()

    for i in range(max_iterations):
        solver_nqueens = GA(size)
        last_pop, all_gen = solver_nqueens.solve(
            npop, ngen, cxpb, mutpb, indpb, verbose=verbose)

        fitness = [item.fitness for item in last_pop]
        if numpy.mean(fitness) == 0.:
            hit_stats["Avg"] += 1
        if numpy.std(fitness) == 0.:
            hit_stats["Std"] += 1
        if numpy.max(fitness) == 0.:
            hit_stats["Max"] += 1
        if numpy.min(fitness) == 0.:
            hit_stats["Min"] += 1

        for j in range(1, len(all_gen)-1):
            stats["Avg"][j - 1] += float(all_gen[j][2]) / max_iterations
            stats["Std"][j - 1] += float(all_gen[j][3]) / max_iterations
            stats["Min"][j - 1] += float(all_gen[j][4]) / max_iterations
            stats["Max"][j - 1] += float(all_gen[j][5]) / max_iterations
        if save is not None:
            data = []
            for ind in last_pop:
                data.append(ind.solution + [ind.fitness])
            df_pop = pandas.DataFrame(
                data=data, columns=[f"allele {i}" for i in range(size)]+["fitness"])
            df_pop.to_excel(os.path.join(save, "population.xlsx"))
            df_log = pandas.DataFrame(data=all_gen)
            df_log.to_excel(os.path.join(save, "log.xlsx"))
    generations = numpy.arange(1, ngen + 1)
    header = ['Generation', 'Minimum Fitness', 'Maximum Fitness',
              'Average Fitness', 'Standard deviation']
    plt.title(r"$N = %d$ , $n_{gen} = %d$ , $n_{pop} = %d$ , $cx_{pb} = %.03f$ , $mut_{pb} = %0.3f$ , $ind_{pb} = %0.3f$" %
              (size, ngen, npop, cxpb, mutpb, indpb))
    plt.plot(generations, stats["Max"], c="blue", label="Max")
    plt.plot(generations, stats["Avg"], c="orange", label="Average")
    plt.plot(generations, stats["Min"], c="grey", label="Min")
    plt.ylabel("Objective Function")
    plt.xlabel("Generation")
    plt.legend(loc="best")

    if save_data is not None:
        dir_name = f"N-{size}_ngen-{ngen}_npop-{npop}_cxpb-{cxpb}_mutpb_{mutpb}_indpb_{indpb}"
        file_name = os.path.join(dir_name, "table.csv")
        data = []
        for i in range(len(generations)):
            data.append([generations[i], round(stats["Min"][i], 3), round(
                stats["Max"][i], 3), round(stats["Avg"][i], 3), round(stats["Std"][i], 3)])
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        df_pop = pandas.DataFrame(data=data, columns=header)
        df_pop.to_csv(file_name, index=False)

        plot_filename = os.path.join(dir_name, "plot.jpg")
        plt.savefig(plot_filename)
        with open(os.path.join(dir_name, "hit_percentage.txt"), "w") as f:
            f.write("Hit Counts :\n")
            f.write(f"\ tMin Hit Count : {hit_stats['Min']}\n")
            f.write(f"\ tAverage Hit Count : {hit_stats['Avg']}\n")
            f.write(f"\ tStandard Dev Hit Count : {hit_stats['Std']}\n")
            f.write(f"\ tMax Hit Count : {hit_stats['Max']}\n\n")
            f.write("Hit Statistics :\n")
            f.write(f"\ tMin Hits : {hit_stats['Min']/max_iterations}\n")
            f.write(f"\ tAverage Hits : {hit_stats['Avg']/max_iterations }\n")
            f.write(
                f"\ tStandard Dev Hit Count : {hit_stats['Std']/max_iterations}\n")
            f.write(f"\ tMax Hits : {hit_stats['Max']/max_iterations }\n\n")

    if print_data:
        # Print table
        tableprint.banner(f" NQUEENS -- Genetic algorithm solver ")
        print(tableprint.header(header, width=20))
        for k in range(len(all_gen) - 2):
            print(tableprint.row([generations[k], stats["Min"][k], stats["Max"]
                  [k], stats["Avg"][k], stats["Std"][k]], width=20), flush=True)

        print()
        # Print statistics on hits
        print(" Hit Counts :")
        print(f"\ tMin Hit Count : { hit_stats[ 'Min']}")
        print(f"\ tAverage Hit Count : {hit_stats['Avg']}")
        print(f"\ tStandard Dev Hit Count : {hit_stats['Std']}")
        print(f"\ tMax Hit Count : {hit_stats['Max']}\n")
        print(" Hit Statistics :")
        print(f"\ tMin Hits : {hit_stats['Min']/ max_iterations}")
        print(f"\ tAverage Hits : {hit_stats['Avg']/ max_iterations}")
        print(f"\ tStandard Dev Hits : {hit_stats['Std']/ max_iterations}")
        print(f"\ tMax Hits : {hit_stats['Max']/max_iterations}")
 # Show plot
        plt.show()


if __name__ == "__main__":
    main()

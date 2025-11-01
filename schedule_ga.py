import random
import pandas as pd

# === Load Data ===
def load_data(path):
    """Load program ratings from CSV file"""
    return pd.read_csv(path)

# === Initialize Population ===
def init_population(programs, pop_size=10):
    """Generate random schedules (permutations of programs)"""
    return [random.sample(programs, len(programs)) for _ in range(pop_size)]

# === Fitness Function ===
def fitness(schedule, ratings):
    """Calculate total rating as fitness value"""
    return sum(ratings[p] for p in schedule)

# === Selection ===
def selection(pop, ratings):
    """Select top two individuals with best fitness"""
    return sorted(pop, key=lambda s: fitness(s, ratings), reverse=True)[:2]

# === Crossover ===
def crossover(parent1, parent2, co_rate):
    """Perform single-point crossover"""
    if random.random() < co_rate:
        point = random.randint(1, len(parent1) - 2)
        child = parent1[:point] + [p for p in parent2 if p not in parent1[:point]]
        return child
    return parent1.copy()

# === Mutation ===
def mutate(schedule, mut_rate):
    """Swap two random positions"""
    if random.random() < mut_rate:
        i, j = random.sample(range(len(schedule)), 2)
        schedule[i], schedule[j] = schedule[j], schedule[i]
    return schedule

# === Genetic Algorithm ===
def genetic_algorithm(data, co_rate=0.8, mut_rate=0.02, generations=50, pop_size=10):
    """Main GA loop"""
    programs = list(data['Program'])
    ratings = dict(zip(data['Program'], data['Rating']))

    population = init_population(programs, pop_size)

    for _ in range(generations):
        new_population = []
        for _ in range(pop_size):
            parents = selection(population, ratings)
            child = crossover(parents[0], parents[1], co_rate)
            child = mutate(child, mut_rate)
            new_population.append(child)
        population = new_population

    best_schedule = selection(population, ratings)[0]
    best_fitness = fitness(best_schedule, ratings)
    return best_schedule, best_fitness

import random
from src.constants import DeepLearningModelType
from src.models.reinforcement_learning.QLearningFunction import QLearning
from src.models.reinforcement_learning.SARSAFunction import SARSA
from src.models.reinforcement_learning.ExpectedSARSAFunction import ExpectedSARSA


def generateSolution():
    initial_pre_setters = [
        generate_random_number(0, 2),  # activationFunction
        generate_random_number(0, 3),  # unitsLSTM
        generate_random_number(0, 1),  # outputLayerActivationFunction
        generate_random_number(0, 3),  # optimizerLSTM
        generate_random_number(0, 2),  # lossFunction
        generate_random_number(0, 3),  # epochs
        generate_random_number(0, 2),  # batch_size
    ]
    solution = {
        "algorithm": generate_random_number(1, 3),  # QLearning , SARSA , ExpectedSARSA
        "preSetters": initial_pre_setters,
        "learning_rate": generate_random_number(0.1, 0.9, 0.1),
        "discount_factor": generate_random_number(0.1, 0.9, 0.1),
        "exploration_prob": generate_random_number(0.1, 0.9, 0.1),
    }
    return solution


def fitnessFunction(
    solution,
    memorySolution,
    memoryAcc,
    memoryPresetters,
    best_fitness,
    deep_learning_model_type,
):
    if isSolutionInside(solution, memorySolution):
        print("solution processed previously so skipping...")
        return memoryAcc[indexOfSolution(solution, memorySolution)], memorySolution

    else:
        memorySolution.append(solution)

        if solution["algorithm"] == 1:
            print("Q-Learning is in progress...")
            best_result, best_result_preSetters = QLearning(
                solution["preSetters"],
                solution["learning_rate"],
                solution["discount_factor"],
                solution["exploration_prob"],
                best_fitness,
                deep_learning_model_type,
            ).run()

        elif solution["algorithm"] == 2:
            print("Sarsa is in progress...")
            best_result, best_result_preSetters = SARSA(
                solution["preSetters"],
                solution["learning_rate"],
                solution["discount_factor"],
                solution["exploration_prob"],
                best_fitness,
                deep_learning_model_type,
            ).run()

        else:
            print("Expected Sarsa is in progress...")
            best_result, best_result_preSetters = ExpectedSARSA(
                solution["preSetters"],
                solution["learning_rate"],
                solution["discount_factor"],
                solution["exploration_prob"],
                best_fitness,
                deep_learning_model_type,
            ).run()

        memoryAcc.append(best_result)
        memoryPresetters.append(best_result_preSetters)
        return best_result, memorySolution


def generate_random_number(start, end, step=1):
    if start > end:
        raise ValueError("Start value must be less than or equal to the end value.")
    if step <= 0:
        raise ValueError("Step size must be greater than zero.")

    # Calculate the number of steps within the range
    num_steps = int((end - start) / step) + 1

    # Generate a random step index
    random_step_index = random.randint(0, num_steps - 1)

    # Calculate the random number
    random_number = start + random_step_index * step

    # If the result is equal to end, return end directly to handle floating-point precision issues
    return min(random_number, end)


def initialize_population(population_size):
    print("Initializing population...")
    population = [generateSolution() for _ in range(population_size)]
    return population


def evaluate_population(
    population,
    memorySolution,
    memoryAcc,
    memoryPresetters,
    best_fitness,
    deep_learning_model_type,
):
    fitness_scores = []
    for solution in population:
        print("Solution {0} is in progress...".format(solution))
        res, memorySolution = fitnessFunction(
            solution,
            memorySolution,
            memoryAcc,
            memoryPresetters,
            best_fitness,
            deep_learning_model_type,
        )
        print("Result for solution is {0}".format(res))
        fitness_scores.append(res)
    return fitness_scores, memorySolution


def select_parents(population, fitness_scores, num_parents):
    if num_parents < 2:
        num_parents = 2
    sorted_indices = sorted(
        range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True
    )
    selected_parents = [population[i] for i in sorted_indices[:num_parents]]
    sorted_fitness_scores = [fitness_scores[i] for i in sorted_indices[:num_parents]]
    return selected_parents, sorted_fitness_scores


def crossover(parent1, parent2):
    # Implement crossover logic to create a new solution
    if parent1["algorithm"] != parent2["algorithm"]:
        print("Crossover of different algorithms...")
        child1 = {
            "algorithm": parent2["algorithm"],
            "preSetters": parent1["preSetters"],
            "learning_rate": parent2["learning_rate"],
            "discount_factor": parent2["discount_factor"],
            "exploration_prob": parent2["exploration_prob"],
        }
        child2 = {
            "algorithm": parent1["algorithm"],
            "preSetters": parent2["preSetters"],
            "learning_rate": parent1["learning_rate"],
            "discount_factor": parent1["discount_factor"],
            "exploration_prob": parent1["exploration_prob"],
        }
        return child1, child2

    else:
        crossover_point = random.randint(1, len(parent1["preSetters"]) - 1)
        print(
            "Crossover of same algorithms with crossover point {0}...".format(
                crossover_point
            )
        )
        child = {
            "algorithm": parent1["algorithm"],
            "preSetters": parent1["preSetters"][:crossover_point]
            + parent2["preSetters"][crossover_point:],
            "learning_rate": (parent1["learning_rate"] + parent2["learning_rate"]) / 2,
            "discount_factor": (parent1["discount_factor"] + parent2["discount_factor"])
            / 2,
            "exploration_prob": (
                parent1["exploration_prob"] + parent2["exploration_prob"]
            )
            / 2,
        }
        return child, None


def are_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False

    for element1, element2 in zip(list1, list2):
        if element1 != element2:
            return False

    return True


def mutate(solution, mutation_rate=0.1):
    # Implement mutation logic to introduce small changes to a solution
    print("mutation in progress...")
    if random.random() < mutation_rate:
        print("mutating preSetters 0...")
        solution["preSetters"][0] = generate_random_number(0, 3)
    if random.random() < mutation_rate:
        print("mutating preSetters 1...")
        solution["preSetters"][1] = generate_random_number(0, 3)
    if random.random() < mutation_rate:
        print("mutating preSetters 2...")
        solution["preSetters"][2] = generate_random_number(0, 1)
    if random.random() < mutation_rate:
        print("mutating preSetters 3...")
        solution["preSetters"][3] = generate_random_number(0, 3)
    if random.random() < mutation_rate:
        print("mutating preSetters 4...")
        solution["preSetters"][4] = generate_random_number(0, 2)
    if random.random() < mutation_rate:
        print("mutating preSetters 5...")
        solution["preSetters"][5] = generate_random_number(0, 3)
    if random.random() < mutation_rate:
        print("mutating preSetters 6...")
        solution["preSetters"][6] = generate_random_number(0, 2)

    if random.random() < mutation_rate:
        print("mutating learning rate...")
        solution["learning_rate"] = generate_random_number(0.1, 0.9, 0.1)
    if random.random() < mutation_rate:
        print("mutating discount factor...")
        solution["discount_factor"] = generate_random_number(0.1, 0.9, 0.1)
    if random.random() < mutation_rate:
        print("mutating exploration probability...")
        solution["exploration_prob"] = generate_random_number(0.1, 0.9, 0.1)
    return solution


def indexOfSolution(solution, solutions):
    for index, s in enumerate(solutions):
        if isSameSolution(solution, s):
            return index
    return -1


def isSolutionInside(solution, solutions):
    return any(isSameSolution(solution, s) for s in solutions)


def isSameSolution(solution1, solution2):
    if solution1["algorithm"] == solution2["algorithm"] and are_lists_equal(
        solution1["preSetters"], solution2["preSetters"]
    ):
        return True
    else:
        return False


def genetic_algorithm(population_size, generations, deep_learning_model_type):
    if generations < 1:
        raise ValueError("Invalid number of generations {0}!".format(generations))

    if population_size < 2:
        raise ValueError("Invalid population size {0}!".format(population_size))

    population = initialize_population(population_size)
    fitness_scores = []
    memorySolution = []
    memoryAcc = []
    memory_pre_setters = []
    best_fitness = [0.0]
    for generation in range(generations):
        print("Generation {0} is in progress...".format(generation))
        fitness_scores, memorySolution = evaluate_population(
            population,
            memorySolution,
            memoryAcc,
            memory_pre_setters,
            best_fitness,
            deep_learning_model_type,
        )

        best_solution_index = fitness_scores.index(max(fitness_scores))
        best_fitness = [fitness_scores[best_solution_index]]

        print(f"Generation {generation + 1} - Best Fitness: {best_fitness[0]}")

        if best_fitness == 100:
            print("Found a solution with fitness 100!")
            break

        selected_parents, sorted_fitness_scores = select_parents(
            population, fitness_scores, num_parents=int(population_size * 0.2)
        )
        print(
            "{0} parents selected out of {1} parents.".format(
                len(selected_parents), int(population_size * 0.2)
            )
        )

        children = []
        while len(children) < population_size - len(selected_parents):
            alternate = generate_random_number(0, 1)
            if alternate == 0:  # CROSSOVER
                index_parent1 = random.choice(range(len(selected_parents)))

                index_parent2 = random.choice(range(len(selected_parents)))
                while (
                    index_parent1 == index_parent2
                ):  # ensure parent1 and parent2 are different solutions
                    index_parent2 = random.choice(range(len(selected_parents)))

                parent1 = selected_parents[index_parent1]
                parent2 = selected_parents[index_parent2]
                child1, child2 = crossover(parent1, parent2)
                children.append(child1)

                if child2 is not None:
                    children.append(child2)
            else:  # MUTATION
                index_parent1 = random.choice(range(len(selected_parents)))
                child = mutate(selected_parents[index_parent1])
                children.append(child)

        print(
            "generated {0} children are being added to the population...".format(
                len(children)
            )
        )
        population = selected_parents + children

    print(
        "GA finished! Population size: {0}, fitness scores size: {1}".format(
            len(population), len(fitness_scores)
        )
    )
    print("Population:\n{0}".format(population))
    print("Fitness scores:\n{0}".format(fitness_scores))
    best_solution_index = fitness_scores.index(max(fitness_scores))
    best_solution_gene = population[best_solution_index]
    best_solution_pre_setters = memory_pre_setters[best_solution_index]
    return best_solution_gene, best_solution_pre_setters


# Example usage
# best_solution, preSetters = genetic_algorithm(
#     population_size=10,
#     generations=3,
#     deep_learning_model_type=DeepLearningModelType.DIFFERENT_SEQUENCES_LSTM,
# )
# print("Best Solution:", best_solution)
# print(preSetters)

# This block is not needed anymore because the models are saved while genetic algorithm is being performed
# It used to be there to run the LSTM model again after finding the best hyper parameters to be able to be saved
#
# LSTM_Model(activationFunction[preSetters[0]], unitsLSTM[preSetters[1]],
#                              outputLayerActivationFunction[preSetters[2]], optimizerLSTM[preSetters[3]],
#                              lossFunction[preSetters[4]], epochs[preSetters[5]], batch_size[preSetters[6]],
#                              [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],True)
# print("Model Saved")

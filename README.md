# Genetic Algorithm TV Scheduling

This project implements a Genetic Algorithm (GA) to optimize television program scheduling. The objective is to generate an efficient TV schedule by assigning programs to available time slots while minimizing scheduling conflicts and maximizing schedule quality.

## Features

* Population-based search using Genetic Algorithm
* Random generation of initial TV schedules
* Fitness evaluation to measure schedule quality
* Selection, crossover, and mutation operations
* Multiple trial execution for performance comparison
* Analysis of fitness scores and convergence behavior

## Methodology

The Genetic Algorithm starts by generating a population of candidate TV schedules. Each schedule is evaluated using a fitness function that considers scheduling constraints and optimization objectives. Through iterative generations, the algorithm applies selection, crossover, and mutation operators to produce improved schedules. The process continues until the maximum number of generations is reached or an optimal solution is found.

## Experiments

Three trials were conducted with different Genetic Algorithm configurations to evaluate performance. The results were compared based on:

* Best fitness value achieved
* Convergence speed
* Schedule quality
* Execution time

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib

## Learning Outcomes

This project demonstrates how evolutionary algorithms can be applied to solve scheduling and optimization problems. It highlights the effectiveness of Genetic Algorithms in exploring large search spaces and generating near-optimal solutions for real-world scheduling applications.


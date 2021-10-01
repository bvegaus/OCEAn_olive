# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:55:52 2020

@author: belen
"""
import random
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, mean_absolute_error
import time
from sklearn.metrics import confusion_matrix
import sys
import os 

# =============================================================================
#  INITIALIZATION
# =============================================================================

def create_initial_population(N, t, s):
    """
        Create the initial population (IP)
    """
    random.seed(s)
    IP = []
    for i in range(N):
        individual = [random.randint(0,100) for i in range(t)]
        IP.append(individual)
    
    return IP

# =============================================================================
#  EVALUATION
# =============================================================================


def fitness(individual, C, l):    
    pred = []
    fitness = 0
    for index, row in C.iterrows():
        union = list(zip(row.tolist(), individual))
        #To create a key-value mmap with the sum of the weights
        label_sum = dict()
        for k, v in union:
            label_sum[k] = label_sum.get(k, 0) + v     

        pred.append(sorted(label_sum.items(), key = lambda x:x[1], reverse =True)[0][0])
    fitness = (abs(pred-l)).sum()
      
    return (fitness, individual)


# =============================================================================
#  TOURNAMENT SELECTION
# =============================================================================
          
def tournament_selection_without_replacement(population_fitness, sel_press):
    random.shuffle(population_fitness)
    lista_parents = []
    chunks = [population_fitness[x:x+sel_press] for x in range(0, len(population_fitness), sel_press)]
    for elem in chunks:
        lista_parents.append(sorted(elem, key=lambda x:x[0])[0][1])

    return lista_parents
        
                
    

# =============================================================================
#  UNIFORM CROSSOVER
# =============================================================================
def uniform_crossover(parent1, parent2):
    child1, child2 = parent1, parent2
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
            
    return child1, child2
    


# =============================================================================
#  MUTATION
# =============================================================================
def mutate(individual, mu_pressure):
    if random.uniform(0.0, 1.0) <= mu_pressure:
        point = random.randint(0, len(individual)-1)
        new_value = random.randint(0,100)
        individual[point] = new_value
    return individual
    



# =============================================================================
#  AUXILIARY FUNCTIONS
# =============================================================================


def sort_population_by_fitness(population_fitness):
    return sorted(population_fitness, key=lambda x:x[0])


def make_population_fitness(population, C, l):
    num_cores = multiprocessing.cpu_count()
    
    return Parallel(n_jobs=num_cores)(delayed(fitness)(individual, C, l) for individual in population)


        
        
        
def make_next_generation_wr(population, N, sel_press, mu_press, C, l):
    next_generation = []
    num_cores = multiprocessing.cpu_count()
    sorted_by_fitness_generation = sort_population_by_fitness(Parallel(n_jobs=num_cores)(delayed(fitness)(individual, C, l) for individual in population))
    next_generation.append(sorted_by_fitness_generation[0][1])

    list_parents = tournament_selection_without_replacement(sorted_by_fitness_generation, sel_press)
    for elem in list_parents:
        next_generation.append(elem.copy())
    ## Selection and Reproduction 
    while len(next_generation) < N-2:
        parent1, parent2 = random.sample(list_parents, 2)
        child1, child2 = uniform_crossover(parent1, parent2)
        next_generation.append(child1)
        next_generation.append(child2)
    if len(next_generation) < N:
        parent1, parent2 = random.sample(list_parents, 2)
        child1, child2 = uniform_crossover(parent1, parent2)
        next_generation.append(child1)


    ##Mutation 
    next_generation_mutated = []
    next_generation_mutated.append(next_generation[0])
    for i in range(1,len(next_generation_mutated)-1):
        next_generation_mutated.append(mutate(next_generation[i], mu_press))          

    return next_generation     
        

def get_predictions(chromosome, C):
    pred = []
    for index, row in C.iterrows():
        union = list(zip(row.tolist(), chromosome))
        label_sum = dict()
        for k, v in union:
            label_sum[k] = label_sum.get(k, 0) + v  
        pred.append(sorted(label_sum.items(), key = lambda x:x[1], reverse =True)[0][0])
    return pred


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mze = 1 - accuracy_score(y_true, y_pred)
    return mae, mze

    

def genetic_algorithm(N, t, s, C, l, G, sel_press, mu_press):
    fitness_evolution = list()

    IP = create_initial_population(N, t, s)
    IP_fitness = make_population_fitness(IP, C, l)
    best_IP = sort_population_by_fitness(IP_fitness)

    fitness_evolution.append(best_IP[0][0])
    
    population = IP.copy()
    
    for g in range(G):
        population = make_next_generation_wr(population, N, sel_press, mu_press, C, l)
        population_fitness = make_population_fitness(population, C, l)
        fitness_evolution.append(sort_population_by_fitness(population_fitness)[0][0])

    print('fitness evolution:     ', fitness_evolution)
    best_cromosome = sort_population_by_fitness(population_fitness)[0][1]
    
    return best_cromosome
    
        


        
        
    

def main():

    N = 200
    t = 18
    G = 40
    sel_press = 2
    mu_press = 0.4
    semilla = [1]
    dataset_name = 'car' # an input parameter between [1-5] which indicate the random partition
    for i in range(5):
        opt = np.str(i+5)
        validation = np.str(i + 10)

        C_opt = pd.read_csv('./predictions/'+dataset_name+'/'+opt+ '.csv')
        l_opt = C_opt.real
        C_opt = C_opt.drop(['real', 'Unnamed: 0'], axis = 1)

        C_validation = pd.read_csv('./predictions/'+dataset_name+'/'+validation+ '.csv')
        l_validation = C_validation.real
        C_validation = C_validation.drop(['real', 'Unnamed: 0'], axis = 1)

        MAE = []
        MZE = []
        for s in semilla:
            print("Semilla", s)
            best_cromosome = genetic_algorithm(N, t, s, C_opt, l_opt, G, sel_press, mu_press)
            print("Best cromosome" , best_cromosome)
            #Validation
            y_true = l_validation.tolist()
            print(y_true)
            y_pred = get_predictions(best_cromosome, C_validation)
            print(y_pred)
            #aux_matrix = confusion_matrix(y_true, y_pred)
            print(confusion_matrix(y_true, y_pred))
            #conf_matrix = conf_matrix + confusion_matrix(y_true, y_pred)
            #print(conf_matrix)
            mae, mze = evaluate(y_true, y_pred)
            MAE.append(mae)
            MZE.append(mze)
            print("\n")

        print("MAE mean:    ", np.mean(MAE))

        print("MZE mean:    ", np.mean(MZE))
    #print(conf_matrix)

        
                             
                             
    
    
    
    



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

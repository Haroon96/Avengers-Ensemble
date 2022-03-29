import pickle
import sys
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import writeprintsStatic as ws
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mlxtend.classifier import EnsembleVoteClassifier
import numpy as np
from sklearn.pipeline import make_pipeline
from EnsembleConstructor import makePipeline
import random
from collections import namedtuple
import util

try:
    datasetName = str(sys.argv[1])
    authorsRequired = int(sys.argv[2])
except:
    datasetName, authorsRequired = 'amt', 5


def saveEnsemble(i, clfs):

    X_train, X_test, y_train, y_test = util.loadData(datasetName, authorsRequired)
    outpath = os.path.join('trainedModels', f'{datasetName}-{authorsRequired}')

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    ensemble = make_pipeline(
        StandardScaler(),
        EnsembleVoteClassifier(clfs, voting='soft')
    )
    
    ensemble.fit(X_train, y_train)
    
    ss = ensemble.named_steps.standardscaler
    
    print("Accuracy", accuracy_score(y_test, ensemble.predict(X_test)))
    print("Ensemble entropy", util.entropy(y_test, util.prediction_matrix(ss.transform(X_test), ensemble.named_steps.ensemblevoteclassifier.clfs_)))
    pickle.dump(ensemble, open(os.path.join(outpath, f'trained_model{i}.sav'), 'wb'))
 

class GeneticAlgorithm:
    def __init__(self, target_entropy, max_iterations, gene_count=10, population_size=10):
        self.target_entropy = target_entropy
        self.max_iterations = max_iterations
        self.gene_count = gene_count
        self.population_size = population_size
        
    # fitness => absolute distance from the target entropy
    def fitness(self, individual):
        return abs(self.target_entropy - util.entropy(y_test, util.prediction_matrix(X_test, individual)))

    # a gene is a classifier
    def gene(self):
        while True:
            clf = makePipeline(total_features)
            clf.fit(X_train, y_train)
            if accuracy_score(y_test, clf.predict(X_test)) >= 0.8:
                break
        return clf

    def select(self, individuals):

        # resample the population
        if random.random() < 0.1:
            _range = range(2, min(len(individuals) + 1, 7))
            individuals = random.sample(individuals, random.choice(_range))

        # return top 2 fittest individuals
        return sorted(individuals, key=self.fitness)[:2]

    def crossover(self, ind1, ind2):
        ind = []

        rand = random.random()
        for i, j in zip(ind1, ind2):
            if rand < 0.45:
                # take gene from first indivdual
                ind.append(i)
            elif rand < 0.9:
                # take gene from second individual
                ind.append(j)
            else:
                # select a new gene
                ind.append(self.gene())

        return ind

    def mutate(self, individual):
        for i in range(len(individual)):
            # mutate the individual
            if random.random() < 0.05:
                individual[i] = self.gene()
        return individual

    def nextgen(self, oldgen):
        newgen = self.select(oldgen)

        # while population isn't complete
        while len(newgen) < 10:
            # select fittest individuals with a degree of randomness
            first, second = self.select(newgen)
            # crossover the two fittest individuals
            individual = self.crossover(first, second)
            # mutate the individual
            individual = self.mutate(individual)
            # add individual to population
            newgen.append(individual)

        return newgen

    # create a new individual
    def initgen(self):
        gen = []
        for i in range(self.population_size):
            gen.append([])
            for j in range(self.gene_count):
                gen[-1].append(self.gene())

        return gen

    def run(self):
        # populate initial generation
        gen = self.initgen()

        for i in range(self.max_iterations):
            # yield this generation
            yield sorted(gen, key=self.fitness)[0]
            
            # evolve new generation
            gen = self.nextgen(gen)
    
def main():
    global X_train, X_test, y_train, y_test, total_features, scaler
    
    X_train, X_test, y_train, y_test = util.loadData(datasetName, authorsRequired)
    total_features = len(X_test[0])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    targets = [1]
    
    for i, target in enumerate(targets):
        print("Target entropy", target)

        ga = GeneticAlgorithm(target, 10)
        for gen_number, gen_best in enumerate(ga.run()):
            ent = util.entropy(y_test, util.prediction_matrix(X_test, gen_best))
            print(f'Generation {gen_number}'.ljust(20), f'Entropy {ent:.2f}'.ljust(50), end='\r')


        for clf in gen_best:
            print("Accuracy", accuracy_score(y_test, clf.predict(X_test)))
            
        print("\nAchieved entropy", f'{ent:.2f}')
        saveEnsemble(i, gen_best)
        print()
      
if __name__ == '__main__':
    main()

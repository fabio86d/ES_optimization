import numpy as np
import random
from operator import itemgetter, attrgetter, methodcaller
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import pylab
from test_functions import generate_3dlandscape, rastring_cost_function, test_function2D


class ES_individual:

    """
        Each individual of an ES population has fields:
        
        - (numpy array) par = parameters set
        - (numpy array) stpar = endogenous strategy parameters
        - (float) fit = fitness value

        An individual can print out his info when requested

    """

    def __init__(self, p, s, f):


        self.par = np.array(p)
        self.stpar = np.array(s)
        if type(f) == float:
            self.fit = f
        else: self.fit = f.astype(float)
    


class EvolutionaryStrategy():

    """ 
        Evolutionary strategy class

        Initialization requires:
        - (sequence of N 2dimensional tuples) domain = (xmin,xmax) for each of the N parameters
        - (int) mu = size of parent population
        - (int) rho = mixing number (#parents involved in the procreation of one offspring) (default value = 1)
        - (int) lambda = size of the offspring population (default value = 7)
        - (sequence of N values) = accuracy limit needed for each dimension

    """


    def __init__(self, objective_function, domain, mu, accuracies, rho = 1, l = 21, var = 1.0, max_numb_generations = 100, selection_type = '(mu+l)', with_plots = True, with_sigma_mutation = True, self_adaptivity = 1.0, with_restart = True):

        # (mu/rho +, l) ES
        self._mu = mu
        self._l = l
        self._rho = rho

        self._lowlim = np.array([x[0] for x in domain])
        self._uplim = np.array([x[1] for x in domain])
        self._objective_function = objective_function
        self._selection_type = selection_type

        # populations parameters
        self._parent_pop = [None]*self._mu
        self._offspring_pop = [None]*self._l
        self._N = len(self._lowlim)
        self._s = np.asarray([var]*self._N)
        
                              
        # mutation parameters
        self._tau = np.divide(1.0,np.sqrt(2*np.sqrt(self._N)))
        self._tau_p = np.divide(1.0,np.sqrt(2*self._N))
        self._constant_strpar = self_adaptivity # if smaller makes the self-adaption of the algorithm slower
        self._with_sigma_mutation = with_sigma_mutation
        self._with_restart = with_restart

        # termination condition parameters
        self._generation_id = 0
        self.max_numb_generations = max_numb_generations
        if len(accuracies) != self._N:
            raise ValueError('Size of accuracies must be the same as the size of domain')
        self._accuracies = self._real_to_norm(np.array(accuracies))
        self._restart = False

        self.solution = ES_individual(np.asarray([]*self._N), np.asarray([]*self._N), self._compute_fitness(np.random.random_sample(size = self._N)))
        self.solution_generation = 0

        print("dim space" , self._N)
    
        # plot parameters
        self._withplots = with_plots
        if self._withplots:
            # plot 1
            self._plotcols = ["blue", "green", "red", "cyan", "yellow", "magenta"]
            self._plot1cols = self._plotcols[:self._mu]
            self.fig = plt.figure()
            self.ax1 = plt.subplot2grid((2,2),(0, 0))
            self.lines1 = []
            for index in range(self._mu):
                lobj = self.ax1.plot([], [], lw=2, color=self._plot1cols[index])[0]
                self.lines1.append(lobj)
            self.ax1.grid()

            self.plot1data = [[] for i in range(self._mu+1)]

            # plot 2
            self.ax2 = plt.subplot2grid((2,2),(0, 1), rowspan = 2)
            self._plot2cols = self._plotcols[:self._N]
            self.scat = self.ax2.scatter([], [], c = self._plot2cols)
            self.x2data, self.y2data = [], []
            
            # plot 3
            self.ax3 = plt.subplot2grid((2,2),(1, 0))
            self._plot3cols = self._plotcols[:self._N]
            self.lines3 = []
            for index in range(self._N):
                lobj = self.ax3.plot([], [], lw=2, color=self._plot3cols[index])[0]
                self.lines3.append(lobj)
            self.plot3data = [[] for i in range(self._N+1)]
            self.ax3.grid()

        # initialize randomly first population within the search domain
        self._initialize()
    
        
    def _initialize(self):

        for i in range(self._mu):

            # generate one normalized random parameters set (samples from uniform distribution in [0,1))
            p = np.random.random_sample(size = self._N)

            # evaluate fitness for the values corresponding to the generated normalized random parameters set
            f = self._compute_fitness(p)

            # assign strategic parameter
            s = self._s

            # generate an individual
            ind = ES_individual(p,s,f)
            #ind.print_info()

            # add to population list
            self._parent_pop[i] = ind

            # add to data plot 2
            if self._withplots:
                self.x2data.extend(ind.par)
                self.y2data.extend([ind.fit]*self._N)            

        # update generation step
        self._generation_id += 1
        print(" \n First random generation initialized ")
        print(" Initial population is ")
        for i in range(self._mu): 
            self._print_info_ind(self._parent_pop[i])

        # for initial plot 1
        if self._withplots:
            self.max_fitness = np.max(self.y2data);
            self.min_fitness = np.min(self.y2data);

        return
    
      
    def _marriage(self, pop):

        """ 
            Selects rho parents from population pop for them to recombine
        
        """

        return random.sample(pop, self._rho)


    def _recombination(self, parents):

        """
            Unlike stadard crossover in GA (where 2 children are born), only one recombinant offspring is typically generated in ES 

        """
        # 2 ways can be implemented: dominant reccombination and intermediate recombination
        # recombination can be performed both on the object parameters and on the endogenous strategy parameters

        # current implementation is the intermediate one for both object parameters and strategy parameters
        # initialize recombinant
        recombinant = ES_individual(np.zeros((self._N)), np.zeros((self._N)), 0.0)
        
        # compute mean values for both object parameters and strategy parameters
        for i in range(self._rho):
            recombinant.par += parents[i].par
            recombinant.stpar += parents[i].stpar
        recombinant.par = np.divide(recombinant.par, self._rho)
        recombinant.stpar = np.divide(recombinant.stpar, self._rho)


        return recombinant


    def _mutation(self, child):

        """
            Mutation for both the object parameters and the strategic parameters or for the object parameters only

        """
        # mutation of strategic parameters 
        if self._with_sigma_mutation:

            ## without limiting the sigma
            #f1 = np.multiply(self._constant_strpar,np.asarray([self._tau_p*np.random.normal()]*self._N))
            #f2 = np.multiply(self._constant_strpar,self._tau*np.random.normal(size = self._N))
            #child.stpar *= np.exp(f1 + f2)

            # by keeping sigma within (0,1) since parameters are normalized
            while True:              

                f1 = np.multiply(self._constant_strpar,np.asarray([self._tau_p*np.random.normal()]*self._N))
                f2 = np.multiply(self._constant_strpar,self._tau*np.random.normal(size = self._N))

                candidate_stpar = np.exp(f1 + f2)

                if self._is_inside_domain(candidate_stpar): 
                    break

            # update child with new accepted strategic parameters
            child.stpar *= candidate_stpar

        else:

            # update child with same strategic parameters
            child.stpar = self._s

        while True:

            # mutate object parameters 
            candidate_par = child.par + np.multiply(child.stpar , np.random.normal(size = self._N))

            # accept if child falls in the search domain
            if self._is_inside_domain(candidate_par): 
                break

        # update child with new accepted object parameters
        child.par = candidate_par

        # update fitness value for child with new accepted object parameters
        self._update_fitness(child)

        return


    def _generate_child(self):
        
        # marriage (parents selection)    
        parents = self._marriage(self._parent_pop)
            
        # recombination between current parents
        child = self._recombination(parents)

        #print "before mutation par" , child.par
        #print "before mutation fit" , child.fit

        # mutation
        self._mutation(child)

        #print "after mutation par" , child.par
        #print "after mutation fit" , child.fit

        return child


    def _selection(self):

        """
            (mu+l) = Plus selection
            Plus selection guarantees the survival of the best individual found so far.
            Since it preserves the best individual such selection techniques are also called
            elitist.

        """

        if self._selection_type == '(mu+l)':

            # update generation step
            self._generation_id += 1

            # define selection pool
            selection_pool = self._parent_pop + self._offspring_pop

            #sort individuals in selection pool from best fitness (lowest value of objective function) to worst fitness (highest value of objective function)
            selection_pool.sort(key=attrgetter('fit'))

            print(" The selection pool is ")
            for a in range(len(selection_pool)): print(selection_pool[a].fit)
            print("  ")

            #print " The parent population before selection "
            #self._print_parent_population_fit()
            #print "  "

            # define new parent population by selecting the first mu individuals of the sorted selection pool
            self._parent_pop = selection_pool[:self._mu]

            #print " The parent population after selection "
            #self._print_parent_population_fit()
            #print "  "

            # empty offspring for next step
            self._offspring_pop = [None]*self._l
            #print " The offspring generation after selection is "
            #self._print_offspring_population_fit()

            print(" Generation", self._generation_id, "completed \n")

            print(" The new parent population is ")
            for i in range(self._mu): self._print_info_ind(self._parent_pop[i])

            return


        if self._selection_type == '(mu,l)':

            assert mu < l

            return


    def _check_for_restart(self):

        stpar_condition = np.zeros((self._mu,self._N))
        for i in range(self._mu):
            for j in range(self._N):
                stpar_condition[i,j] = False if self._parent_pop[i].stpar[j] > self._accuracies[j] else True 
            
        if np.all(stpar_condition):
            
            # restart needed
            # assign temporary solution
            if self._parent_pop[0].fit < self.solution.fit:
                self.solution = ES_individual(self._norm_to_real(self._parent_pop[0].par), self._parent_pop[0].stpar, self._parent_pop[0].fit)
                self.solution_generation = self._generation_id

            # generate 2 new parents
            for i in range(self._mu):

                # generate one normalized random parameters set (samples from uniform distribution in [0,1))
                p = np.random.random_sample(size = self._N)

                # evaluate fitness for the values corresponding to the generated normalized random parameters set
                f = self._compute_fitness(p)

                # assign strategic parameter
                s = self._s

                # generate an individual
                ind = ES_individual(p,s,f)
                #ind.print_info()

                # add to population list
                self._parent_pop[i] = ind

        return


    def run(self):
       
        if not self._withplots:

            while not self._termination_condition():

                # generate offspring of l children (recombined and mutated from parents population)
                print(" \n Generation of new offspring \n")
                for i in range(self._l):

                    self._offspring_pop[i] = self._generate_child()
                    self._print_info_ind(self._offspring_pop[i])

                #print " "
                #self._print_offspring_population_fit()
                #print " " 

                # selection
                print("\n Selection \n")
                self._selection()

                if self._with_restart:
                    self._check_for_restart()

                # plot
                #self._plot2D_parameters(0,1)  

        else:

            # the combination of blit = True and interval = 10ms makes the code run much faster
            ani = animation.FuncAnimation(self.fig, self._anim_update_line, self._compute_data, blit=True, interval=10,
                                            init_func=self._anim_init)     

            plt.show()


        # assign solution
        if self._parent_pop[0].fit < self.solution.fit:
            self.solution = ES_individual(self._norm_to_real(self._parent_pop[0].par), self._parent_pop[0].stpar, self._parent_pop[0].fit)              
            self.solution_generation = self._generation_id

        return 


    def _compute_data(self):

        """ core of the algorithm: returns a generator, used as input to the anim_update_line function called for the animation """

        while not self._termination_condition():

            # generate offspring of l children (recombined and mutated from parents population)
            print(" \n Generation of new offspring \n")
            for i in range(self._l):

                self._offspring_pop[i] = self._generate_child()
                self._print_info_ind(self._offspring_pop[i])

                # add offspring data to data plot 2
                self.x2data.extend(self._offspring_pop[i].par)
                self.y2data.extend([self._offspring_pop[i].fit]*self._N)

            #print " "
            #self._print_offspring_population_fit()
            #print " " 

            # selection
            print("\n Selection \n")
            self._selection()

            if self._with_restart:
                self._check_for_restart()

            # plot
            #self._plot2D_parameters(0,1)  

            yield self._generation_id


    def _compute_fitness(self, p):

        """ computes fitness given a set of normalized parameters """

        return self._objective_function(self._norm_to_real(p))


    def _update_fitness(self, individual):

        """ compute fitness corresponding to the current parameters given the individual """

        individual.fit = self._compute_fitness(individual.par)

        return

    def _termination_condition(self):

        # resources based
        if self._generation_id > self.max_numb_generations:
            return True 

        return False


    def _is_inside_domain(self, p):

        """ returns True if all normalized parameters in the given array are inside [0,1)"""
       
        return np.all(p >= 0) and np.all(p < 1)


    def _norm_to_real(self, p):

        """ converts normalized parameter set to real values """

        return np.multiply((self._uplim - self._lowlim),p) + self._lowlim

    def _real_to_norm(self, p):

        """ converts real parameter set to normalized values in the range (0,1)"""

        return np.divide(p, (self._uplim - self._lowlim))


    def _anim_init(self):

        """ function for initialization of fitness plot """

        self.ax1.set_xlim(0, self.max_numb_generations/5)
        self.ax1.set_ylim(self.min_fitness, self.max_fitness)
        self.ax1.set_xlabel("Generation ID")
        self.ax1.set_ylabel("Parents Population Fitness")

        self.ax2.set_xlabel("Normalized Parameters")
        self.ax2.set_ylabel("Fitness")
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(self.min_fitness, self.max_fitness)

        self.ax3.set_xlabel("Generation ID")
        self.ax3.set_ylabel("Strategic param (first individual only)")
        self.ax3.set_xlim(0, 200)
        self.ax3.set_ylim(0, 1)

        for line in self.lines1:
            line.set_data([],[])

        for line in self.lines3:
            line.set_data([],[])

        self.scat.set_offsets([])
        #self.scat.set_array()

        return tuple(self.lines1) + (self.scat,) + tuple(self.lines3)


    def _anim_update_line(self, id):

        """ anim_update_line function called for the animation """

        # add new data plot to lines
        self.plot1data[0].append(id) 
        self.plot3data[0].append(id) 
               
        for i in range(self._mu):
            self.plot1data[i+1].append(self._parent_pop[i].fit)            

            # for plot 1
            if self._parent_pop[i].fit > self.max_fitness : self.max_fitness = self._parent_pop[i].fit
            if self._parent_pop[i].fit < self.min_fitness : self.min_fitness = self._parent_pop[i].fit

        for lnum,line in enumerate(self.lines1):
            line.set_data(self.plot1data[0], self.plot1data[lnum+1])

        for i in range(self._N):
            self.plot3data[i+1].append(self._parent_pop[0].stpar[i])
        
        for lnum,line in enumerate(self.lines3):
            line.set_data(self.plot3data[0], self.plot3data[lnum+1])

        # for plot 1
        xmin, xmax = self.ax1.get_xlim()
        if id > xmax: 
            self.ax1.set_xlim(0, 2*xmax)
            self.ax1.figure.canvas.draw() # this function should be called as little as possible       
        self.ax1.set_ylim(self.min_fitness -5, self.max_fitness)

        # for plot 2
        self.scat.set_offsets(np.asarray((self.x2data, self.y2data)).T)
        self.ax2.set_ylim(self.min_fitness -5, self.max_fitness)

        #plt.draw()

        return tuple(self.lines1) + (self.scat,) + tuple(self.lines3)


    def _plot2D_parameters(self, par1, par2):

        """ 2D plot of the current population in the search domain relative to two parameters with indices par1,par2 amoung the N ones """

        assert par1 < self._N
        assert par2 < self._N
        
        # only for this particular case
        plot2D_cost_function(Z, [self._lowlim[par1], self._lowlim[par2]], [self._uplim[par1], self._uplim[par2]])

        # real function
        for x in range(self._mu):
            plt.scatter(self._norm_to_real(self._parent_pop[x].par)[par1],self._norm_to_real(self._parent_pop[x].par)[par2], marker = 'o', s = 50)       
        plt.xlim((self._lowlim[par1] , self._uplim[par1] ))
        plt.ylim((self._lowlim[par2] , self._uplim[par2] ))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()



    def _print_info_ind(self, ind):

        """ Prints out info about the given individual """

        print("I am a new individual")
        print("My parameters set is ", self._norm_to_real(ind.par))
        print("My fitness value is ", ind.fit)
        print("My strategic parameter is ", ind.stpar)


    def _print_parent_population_fit(self):

        if self._parent_pop.count(None) == len(self._parent_pop):
            print(" Current parent population is empty ")        
        else:
            print(" Current parent population has fitness values:")
            for a in range(self._mu): print(self._parent_pop[a].fit) 


    def _print_offspring_population_fit(self):

        if self._offspring_pop.count(None) == len(self._offspring_pop):
            print(" Current offspring population is empty " )          
        else:
            print(" Current offspring population has fitness values:")
            for a in range(self._l): print(self._offspring_pop[a].fit) 

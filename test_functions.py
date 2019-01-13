import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# this script includes some multi-dim test functions for optimization and a function to generate 3D plot of the landscape for two chosen dimensions

def rosenbrock_cost_function(x):

    """ 
    The globabl minimum lays inside a long, narrow, parabolic shaped flat valley.
    To find the valley is trivial, however convergence to the global optimum is difficult.
    For N = 3 the global minimum f(x) = 0 is at x = (1,1,1). For N between 4 and 7 there is another local minimum.
    Domain is usually [-2.048, 2.048] """

    n = len(x)
    output = 0
    
    for i in range(n-1):
        output += (100*np.square(x[i+1] - (x[i]**2)) + np.square(1 - x[i]))

    return output



def rastring_cost_function(x):

    """
    Test function with many local minima (provided by modulation with sin) and one global minimum.  
    The dim is given by the given array in input, indicating the point where the function needs to be evalauted. 
    Global minimum in the origin. Adviced domain is usually [-5.12, 5.12] for all dimensions """

    n = len(x)

    output = 10*n
    
    for i in range(n):
        output += np.square(x[i]) - 10*np.cos(2*np.pi*x[i])
     

    return output


def test_function2D(p):

    """ 
    2D test function with global minimum in the origin and local minima in the form of rings aorund the global minimum"""

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    def g(r):

        return (r**2./(r+1.))*(np.sin(r) - (np.sin(2.*r)/2.) + (np.sin(3.*r)/3.) - (np.sin(4.*r)/4.) + 4.)

    def h(t):

        return 2. + np.cos(t) + ((np.cos(2.*t - 0.5)/2.))

    (rho, phi) = cart2pol(p[0], p[1])

    return g(rho)*h(phi)


def ackley_function(x):

    """ 
    Deep global minimum in the origin with many small local minima around the hole. Domain [-32.768, 32.768] """

    a = 20
    b = 0.2
    c = 2*np.pi

    n = len(x)

    f1 = 0
    f2 = 0

    for i in range(n):
        f1 += np.square(x[i])
        f2 += np.cos(c*x[i])

    output = -a*(np.exp(-b*np.sqrt(np.divide(f1,n)))) - np.exp(np.divide(f2,n)) + a + np.exp(1)
    
    return output 


def generate_3dlandscape(cost_function, domain, dim1, dim2, npoints):

    """ 
    This function plots a 3D landscape with N points evaluated 
    from a given a N-dim cost function, a domain as an N array of [lowlim,uplim] 
    and the two dimensions to be plotted (starting from 1, not zero!)"""

    dimx = dim1-1
    dimy = dim2-1
    N = len(domain)
    values = np.zeros((npoints,npoints))
    otherdims = (i for i in range(N) if (i != dimx and i != dimy))
    v = np.empty(N)

    # generate domain space
    drx = np.linspace(domain[dimx][0], domain[dimx][1], npoints)
    dry = np.linspace(domain[dimy][0], domain[dimy][1], npoints)

    for i in otherdims:             
            v[i] = np.divide(domain[i][1] + domain[i][0], 2)   
           
    #for x in product(*dr):
    for i in range(npoints):

        v[dimx]= drx[i]
        
        for j in range(npoints):

            v[dimy]= dry[j]

            values[i,j] = cost_function(v)

    # plots
    X,Y = np.meshgrid(drx,dry)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, values ,rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()

    return values


def plot_optimization_result2D(landscape, domain, solution, dim1, dim2):
    
    plt.imshow(landscape,  extent= [domain[dim1-1][0], domain[dim1-1][1], domain[dim2-1][0], domain[dim2-1][1]])

    plt.scatter(solution[dim1-1],solution[dim2-1], marker = 'o', s = 50)       
    #plt.xlim((domain[0][0] , domain[0][1] ))
    #plt.ylim((domain[1][0] , domain[1][1] ))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return



if __name__ == "__main__":

    #domain = [[-5.12, 5.12],[-5.12, 5.12],[-5.12, 5.12],[-5.12, 5.12],[-5.12, 5.12],[-5.12, 5.12]]
    #generate_3dlandscape(rastring_cost_function, domain, 1, 2, 30)

    domain = [[-40,40],[-40,40]]
    generate_3dlandscape(test_function2D, domain, 1, 2, 30)
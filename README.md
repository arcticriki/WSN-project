# Fountain Codes Based Distributed Storage Algorithms for Wireless Sensor Networks

This repository contains the code for the final project of the 2016/2017 course in Wireless Communications. The project consists in studying and implementing some Fountain Codes Based algorithms and testing their performances in a WSN simulator.
The implemented algorithms can be found at these links:
http://ieeexplore.ieee.org/document/4215776/ [1]
http://ieeexplore.ieee.org/document/4505472/ [2]

## Requirements
- Any Operating systems (tested in Windows 10, and MACOS X 10.11.6)
- Python 2.7. For used libraries, see source code

## Running the algorithms
Each file is representative of one developed algorithm. In *Dati* folder is present some auxiliary data used for EDFC and the *Mathematica* code used to solve the optimization problem.
The WSN is created as a *Random Geometric Graph (RGG)*. An exemple of RGG can be found in ```plot_grafo.py```. Sensor nodes are objects of class *Sensor*, which extends the more general class *Storage*. In the file ```Node.py``` can be found all the specifications about the two classes.
All the algorithms are differentied in how they disseminate, collect, encode and store packets, but all of them use ```message_passing.py``` to decode data.
In ```RSD.py``` is present a python implementation of the Robust Soliton Distribution generation and sampling. 

## EDFC
Simply run  ```python EDFC.py ``` for the first algorithm in [1].
In the main you can find:

    ```comparison_network_dimensions(iteration_to_mediate, L_RW, mp): ```
    plot decoding performances for different network deployments, specified in main. Choose the number of iterations to mediate, the random walk length and the number of message passings.
    
    ```comparison_solution_opt_problem(iteration_to_mediate, L_RW, n0, k0, mp)```
    plot decoding performances changing the solution for redundancy coefficients.  Choose the number of iterations to mediate, the random walk length, number of 
    nodes and sesnors and message passings.
    
    ```comparison_length_of_random_walk(iteration_to_mediate, punti, n0, k0, mp)``
    Comparison for several length of the random walk. Choose the number of iterations to mediate, number of nodes and sesnors and message passings.

    ```dissemination_cost(iteration_to_mediate)```
    plot the time complexity of the algorithm



## LTCDS-I
Simply run ```LTCDS-I.Py``` for the first algorithm in [2].
In the main you can find:

    ```comparison network dimensions(iteration_to_ mediate,C1,MP)```
    plot decoding performances for different network deployments, specified in main. Choose the number of iterations to mediate, C1 and 
    the number of message passings.
  
    ```comparison_C1(iteration_to_mediate, MP)```
    plot the system beahviour for several values of system parameter C1. Choose the number of iterations to mediate and message passings

    ```figure_5(iteration_to_mediate, MP)```
    plot figure 5 of [2]. Choose the number of iterations to mediate and message passings

    ```dissemination_cost(iteratin_to_mediate)```
    plot the time complexity of the algorithm

## LTCDS - II
Simply run ```LTCDS-II.py```, for the second algorithm in [2].
In the main you can find:

    ```comparison network dimensions(iteration_to_ mediate, C2, C3, MP)```
    plot decoding performances for different network deployments, specified in main. Choose the number of iterations to mediate, C1, C2 and 
    the number of message passings.
  
    ```comparison_C2(iteration_to_mediate, MP)```
    plot the system beahviour for several values of system parameter C2. Choose the number of iterations to mediate and message passings

    ```comparison_C3(iteration_to_mediate, MP)```
    plot the system beahviour for several values of system parameter C3. Choose the number of iterations to mediate and message passings

    ```dissemination_cost(iteratin_to_mediate)```
    plot the time complexity of the algorithm

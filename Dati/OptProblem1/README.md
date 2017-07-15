# Optimization Problem

To solve it we have exploited the *Constrained and Unconstrained Local and Global Optimization* library of the *Mathematica*.

## How To Use The Code
The code contains a routine that computes the PDF of Robust Soliton Distribution fixing the number of sources *K*, the maximum packet error rate *Delta* and the constant *c*. All these parameters can be chosen by the user.

The chosen solver is **Simulated Annealing**. 
The setting of objective function and constraints follows what presented in [1].
It is possible to set the Violation Probabilities or the Bounds on them  commenting/uncommenting the function named *cf*.
It is possible to save the results to a CSV file uncommenting the *Export* function.
    
    

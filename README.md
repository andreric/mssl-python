# mssl-python

> **Under development as of May 24, 2018** 


Multi-task Sparse Structure Learning (MSSL) method in Python


This repository contains a Python implementation of the MSSL algorithm proposed in the paper <a href="http://jmlr.org/papers/v17/15-215.html">Multi-task Sparse Structure Learning with Gaussian Copula Models</a>.


# Overview #
MSSL belongs to the realm of Multi-task Learning algorithms. Given a set of tasks (regression or classification problems) MSSL learns the parameter set for all tasks in such a way that exploits the structural dependencies among them to improve the performance of individual tasks. Unlike many other multi-task learning methods, MSSL does not assume any type of structure beforehand, but learn it from the data. To learn both task parameters for all tasks and relatedness information MSSL uses an alternating minimization algorithm.

# How to run it? #
A file called *run_demo.py* was created to show how to run MSSL code. In this example tasks are synthetic regression problems. From this script one can change it as needed to deal with classification problems.

~~MSSL has two main steps: (1) learn the tasks specific weights (W-step), and (2) relatedness structure (Omega-step). For the W-step four methods are available to use: closed-form, fminunc (many optimization algorithms implemented in MATLAB through fminunc function), L-BFGS, and FISTA. To use L-BFGS, an external library is required. In our experiments we used the L-BFGS-B mex wrapper, a MATLAB wrapper for the well known fortran code wrote by Prof. Jorge Nocedal and colleagues.
Structure of the input data files
In order to run the code the input data files containing the training and test data must follow a specific format. The alternating_minimization() function, which is the core of MSSL, receives two arrays of cells, X (design matrices) and Y (outputs), with both having the length equal to the number of tasks K. In the X array, each cell is a matrix n x d corresponding to the input data of a specific task. The same happens for the array Y, except that now each cell is a vector of size n.~~

# How to cite it? #
If you like it and want to cite it in your papers, you can use the following:
```
@article{Goncalves2016,
  author  = {Andr{{\'e}} R. Gon\c{c}alves and Fernando J. Von Zuben and Arindam Banerjee},
  title   = {Multi-task Sparse Structure Learning with Gaussian Copula Models},
  journal = {Journal of Machine Learning Research},
  year    = {2016},
  volume  = {17},
  number  = {33},
  pages   = {1-30}
}
```

# Have a question? #
If you found any bug or have a question, don't hesitate to contact me at: andre -at- cs -dot- umn -dot- edu

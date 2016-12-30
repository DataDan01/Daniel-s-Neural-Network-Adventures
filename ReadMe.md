
# Daniel's Neural Network Adventures ReadMe

### Overview

The purpose of this GitHub repository is to document my adventures in building my first neural network. My goal is to build a simple multi-layered network and train it using stochastic gradient descent. I will build out all of the R code by hand to get a better appreciation for the mechanics of neural networks. 

### The Plan

1. Find and prepare a data set to work on (Done)
2. Build a single neuron and figure out how to train it (Done)
3. Chain together a few neurons by hand, train them, and see if they perform better than the single neuron (Done)
4. Create an automated way to chain together neurons (In Progress)

### Current Progress

I'm getting stuck while trying to roll up neurons into each other and finally into the sigmoid output. I successfully programmatically rolled everything into one massive sigmoid function and was able to compute the partial derivatives. For the single neuron, playing around with the learning rate and capping the partial derivatives (to prevent sigmoid gradient explosion) led to reasonable accuracy. Stochastic gradient descent on the deeper network is creating weights that basically output a probability of 0.5, which leads to a similar log-loss as just naively guessing the average. This was a decent first attempt but I am definitely missing some intuition here. The current plan is to [learn more calculus](https://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-calculus-fall-2010) so I can [learn more about neural networks](http://cs231n.stanford.edu/). I already picked up some Python to prepare for CS231n. 

### Bonus Goals

1. Experiment with different types of training approaches and cost functions
2. Automatically create "random" networks, train them, and ensemble them to create a meta-model:
  + Random activation functions
  + Random cost functions
  + Random connections
3. Try to build a faster implementation with Rcpp or Julia

### Resources

[Hacker's guide to Neural Networks by Andrej Karpathy](http://karpathy.github.io/neuralnets/)

[A Step by Step Backpropagation Example by Matt Mazur](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

[Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)

Regardless of my success with this project, I want to thank the above authors for their hard work. I'm also extending this thanks to the countless resources I will use that won't be mentioned here. Any failure here is my own. These guides are well written and I'm happy that I have access to such material for free.

Thanks for browsing!

Daniel Alaiev

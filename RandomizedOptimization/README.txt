Sol Won

#WHY?
The purpose of this project is to explore random search. As always, it is important to realize that understanding an algorithm or technique requires more than reading about that algorithm or even implementing it. One should actually have experience seeing how it behaves under a variety of circumstances.

As such, you will be asked to implement several randomized search algorithms. In addition, you will be asked to exercise your creativity in coming up with problems that exercise the strengths of each.

As always, you may program in any language that you wish insofar as you feel the need to program. As always, it is your responsibility to make sure that we can actually recreate your narrative if necessary.

#PROBLEMS
You must implement four local random search algorithms. They are:
randomized hill climbing
simulated annealing
a genetic algorithm
MIMIC

In addition to analyzing discrete optimization problems, you will also use the first three algorithms to find good weights for a neural network. In particular, you will use them instead of backprop for the neural network you used in assignment #1 on at least one of the problems you created for assignment #1. Notice that this assignment is about an optimization problem and about supervised learning problem. That probably means that looking at only the loss or only the accuracy won’t tell you the whole story. Luckily, you have already learned how to write an analysis on optimization problems and on supervised learning problems; now you just have to integrate your knowledge.

Because we are nice, we will also let you know about some pitfalls you might run into:

The weights in a neural network are continuous and real-valued instead of discrete so you might want to think a little bit about what it means to apply these sorts of algorithms in such a domain.
There are different loss and activation functions for NNs. If you use different libraries across your assignments, you need to make sure those are the same. For example, if you used scikit-learn and don’t modify the ABAGAIL example, they are not.

#HOW?
Instructions on running the code (all the codes are written in jupyter notebook)
- Open anaconda navigator (anaconda 3) and select 'Launch' unnder Jupyter Notebook
- Navigate to the folder downnloaded - CS7641 Randomized Optimization - 
	Part 1 - open 'Randomized_Optimization_4_peaks.ipynb'
	Part 1 - open 'Randomized_Optimization_ContinuousPeaks.ipynb'
	Part 1 - open 'Randomized_Optimization_Knapsack.ipynb'
	Part 2 - open 'Randomized_Optimization_OptimalWeight.ipynb'
	(there should be 4 .ipynb files and 1 .csv files)
- Run it!
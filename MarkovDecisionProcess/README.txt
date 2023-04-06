Sol Won

#WHY
In some sense, we have spent the semester thinking about machine learning techniques for various forms of function approximation. It's now time to think about using what we've learned in order to allow an agent of some kind to act in the world more directly. This assignment asks you to consider the application of some of the techniques we've learned from reinforcement learning to make decisions.

The same ground rules apply for programming languages as with the previous assignments.

#PROBLEMS
You are being asked to explore Markov Decision Processes (MDPs):
Come up with two interesting MDPs. Explain why they are interesting. They don't need to be overly complicated or directly grounded in a real situation, but it will be worthwhile if your MDPs are inspired by some process you are interested in or are familiar with. It's ok to keep it somewhat simple. For the purposes of this assignment, though, make sure one has a "small" number of states, and the other has a "large" number of states. I'm not going to go into detail about what large is, but 200 is not large. Furthermore, because I like variety no more than one of the MDPs you choose should be a so-called grid world problem.
Solve each MDP using value iteration as well as policy iteration. How many iterations does it take to converge? Which one converges faster? Why? How did you choose to define convergence? Do they converge to the same answer? How did the number of states affect things, if at all?
Now pick your favorite reinforcement learning algorithm and use it to solve the two MDPs. How does it perform, especially in comparison to the cases above where you knew the model, rewards, and so on? What exploration strategies did you choose? Did some work better than others?

#HOW?
Instructions on running the code (all the codes are written in jupyter notebook)
- Open anaconda navigator (anaconda 3) and select 'Launch' unnder Jupyter Notebook
- Navigate to the folder downnloaded - CS7641 Markov Decision Process - 
	Part 1 - open 'MDP_largeProblem.ipynb'
	Part 2 - open 'MDP_smallProblem.ipynb'
- Run it!
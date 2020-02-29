#!/usr/bin/env python
# coding: utf-8

# # Constraint Satisfaction Problems
# ---
# Constraint satisfaction is a general problem solving technique for solving a class of combinatorial optimization problems by imposing limits on the values in the solution. The goal of this exercise is to practice formulating some classical example problems as constraint satisfaction problems (CSPs), and then to explore using a powerful open source constraint satisfaction tool called [Z3](https://github.com/Z3Prover/z3) from Microsoft Research to solve them. Practicing with these simple problems will help you to recognize real-world problems that can be posed as CSPs; some solvers even have specialized utilities for specific types of problem (vehicle routing, planning, scheduling, etc.).
# 
# There are many different kinds of CSP solvers available for CSPs. Z3 is a "Satisfiability Modulo Theories" (SMT) solver, which means that unlike the backtracking and variable assignment heuristics discussed in lecture, Z3 first converts CSPs to satisfiability problems then uses a [boolean satisfiability](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem) (SAT) solver to determine feasibility. Z3 includes a number of efficient solver algorithms primarily developed to perform formal program verification, but it can also be used on general CSPs. Google's [OR tools](https://developers.google.com/optimization/) includes a CSP solver using backtracking with specialized subroutines for some common CP domains.
# 
# ## I. The Road Ahead
# 
# 0. [Cryptarithmetic](#I.-Cryptarithmetic) - introducing the Z3 API with simple word puzzles
# 0. [Map Coloring](#II.-Map-Coloring) - solving the map coloring problem from lectures
# 0. [N-Queens](#III.-N-Queens) - experimenting with problems that scale
# 0. [Revisiting Sudoku](#IV.-Revisiting-Sudoku) - revisit the sudoku project with the Z3 solver

# <div class="alert alert-box alert-info">
# NOTE: You can find solutions to this exercise in the "solutions" branch of the git repo, or on GitHub [here](https://github.com/udacity/artificial-intelligence/blob/solutions/Exercises/1_Constraint%20Satisfaction/AIND-Constraint_Satisfaction.ipynb).
# </div>

# In[4]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from util import displayBoard
from itertools import product
# from IPython.display import display

__builtins__.Z3_LIB_DIRS = ['/home/workspace/z3/bin']
from z3 import *


# ---
# ## I. Cryptarithmetic
# 
# We'll start by exploring the Z3 module with a _very_ simple & classic CSP problem called cryptarithmetic. A cryptarithmetic puzzle is posed as an arithmetic equation made up of words where each letter represents a distinct digit in the range (0-9). (This problem has no practical significance in AI, but it is a useful illustration of the basic ideas of CSPs.) For example, consider the problem and one possible solution shown below:
# 
# ```
#   T W O  :    9 3 8
# + T W O  :  + 9 3 8
# -------  :  -------
# F O U R  :  1 8 7 6
# ```
# There are six distinct variables (F, O, R, T, U, W), and when we require each letter to represent a disctinct number (e.g., F != O, R != T, ..., etc.) and disallow leading zeros (i.e., T != 0 and F != 0) then one possible solution is (F=1, O=8, R=6, T=9, U=7, W=3). 
# 
# ### IMPLEMENTATION: Declaring Variables
# For this problem we need a single variable for each distinct letter in the puzzle, and each variable will have an integer values between 0-9. (We will handle restricting the leading digits separately.) Complete the declarations in the next cell to create all of the remaining variables and constraint them to the range 0-9.

# In[6]:


ca_solver = Solver()  # create an instance of a Z3 CSP solver

F = Int('F')  # create an z3.Int type variable instance called "F"
ca_solver.add(0 <= F, F <= 9)  # add constraints to the solver: 0 <= F <= 9
# ...
# TODO: Add all the missing letter variables
O = Int('O')
ca_solver.add(0 <= O, O <= 9)
R = Int('R')
ca_solver.add(0 <= R, R <= 9)
T = Int('T')
ca_solver.add(0 <= T, T <= 9)
U = Int('U')
ca_solver.add(0 <= U, U <= 9)
W = Int('W')
ca_solver.add(0 <= W, W <= 9)


# ### IMPLEMENTATION: Encoding Assumptions as Constraints
# We had two additional assumptions that need to be added as constraints: 1) leading digits cannot be zero, and 2) no two distinct letters represent the same digits. The first assumption can simply be added as a boolean statement like M != 0. And the second is a _very_ common CSP constraint (so common, in fact, that most libraries have a built in function to support it); z3 is no exception, with the Distinct(var_list) constraint function.

# In[7]:


# TODO: Add constraints prohibiting leading digits F & T from taking the value 0
ca_solver.add(T!=0)
ca_solver.add(F!=0)

# TODO: Add a Distinct constraint for all the variables
ca_solver.add(Distinct(F, O, R, T, U, W))


# ### Choosing Problem Constraints
# There are often multiple ways to express the constraints for a problem. For example, in this case we could write a single large constraint combining all of the letters simultaneously $T\times10^2 + W\times10^1 + O\times10^0 + T\times10^2 + W\times10^1 + O\times10^0 = F\times10^3 + O\times10^2 + U\times10^1 + R\times10^0$. This kind of constraint works fine for some problems, but large constraints cannot usually be evaluated for satisfiability unless every variable is bound to a specific value. Expressing the problem with smaller constraints can sometimes allow the solver to finish faster.
# 
# For example, we can break out each pair of digits in the summands and introduce a carry variable for each column: $(O + O)\times10^0 = R\times10^0 + carry_1\times10^1$ This constraint can be evaluated as True/False with only four values assigned.
# 
# The choice of encoding on this problem is unlikely to have any effect (because the problem is so small), however it is worth considering on more complex problems.
# 
# ### Implementation: Add the Problem Constraints
# Pick one of the possible encodings discussed above and add the required constraints into the solver in the next cell. 

# In[8]:


# TODO: add any required variables and/or constraints to solve the cryptarithmetic puzzle
# Primary solution using single constraint for the cryptarithmetic equation
c1 = Int('C1')
c2 = Int('C2')

ca_solver.add(0<=c1, c1<=9)
ca_solver.add(0<=c2, c2<=9)

ca_solver.add(O+O==R+c1*10)
ca_solver.add(c1+W+W==U+c2*10)
ca_solver.add(c2+T+T==O+F*10)


# In[9]:


assert ca_solver.check() == sat, "Uh oh...the solver did not find a solution. Check your constraints."
print("  T W O  :    {} {} {}".format(ca_solver.model()[T], ca_solver.model()[W], ca_solver.model()[O]))
print("+ T W O  :  + {} {} {}".format(ca_solver.model()[T], ca_solver.model()[W], ca_solver.model()[O]))
print("-------  :  -------")
print("F O U R  :  {} {} {} {}".format(ca_solver.model()[F], ca_solver.model()[O], ca_solver.model()[U], ca_solver.model()[R]))


# ### Cryptarithmetic Challenges
# 0. Search online for [more cryptarithmetic puzzles](https://www.reddit.com/r/dailyprogrammer/comments/7p5p2o/20180108_challenge_346_easy_cryptarithmetic_solver/) (or create your own). Come to office hours or join a discussion channel to chat with your peers about the trade-offs between monolithic constraints & splitting up the constraints. (Is one way or another easier to generalize or scale with new problems? Is one of them faster for large or small problems?)
# 0. Can you extend the solution to handle complex puzzles (e.g., using multiplication WORD1 x WORD2 = OUTPUT)?

# ---
# ## II. Map Coloring
# 
# [Map coloring](https://en.wikipedia.org/wiki/Map_coloring) is a classic example of CSPs. A map coloring problem is specified by a set of colors and a map showing the borders between distinct regions. A solution to a map coloring problem is an assignment of one color to each region of the map such that no pair of adjacent regions have the same color.
# 
# Run the first cell below to declare the color palette and a solver. The color palette specifies a mapping from integer to color. We'll use integers to represent the values in each constraint; then we can decode the solution from Z3 to determine the color applied to each region in the map.
# 
# ![Map coloring is a classic example CSP](map.png)

# In[42]:


# create instance of Z3 solver & declare color palette
mc_solver = Solver()
colors = {'0': "Blue", '1': "Red", '2': "Green"}


# ### IMPLEMENTATION: Add Variables
# Add a variable to represent each region on the map above. Use the abbreviated name for the regions: WA=Western Australia, SA=Southern Australia, NT=Northern Territory, Q=Queensland, NSW=New South Wales, V=Victoria, T=Tasmania. Add constraints to each variable to restrict it to one of the available colors: 0=Blue, 1=Red, 2=Green.

# In[43]:


def create_var(name=None, min_val=None, max_val=None):
    var = Int(name)
    return var, min_val <= var, var <= max_val

# create variables and its range of color values
WA, min_cons, max_cons = create_var("WA", 0, 2)
mc_solver.add(min_cons, max_cons)

SA, min_cons, max_cons = create_var("SA", 0, 2)
mc_solver.add(min_cons, max_cons)

NT, min_cons, max_cons = create_var("NT", 0, 2)
mc_solver.add(min_cons, max_cons)

Q, min_cons, max_cons = create_var("Q", 0, 2)
mc_solver.add(min_cons, max_cons)

NSW, min_cons, max_cons = create_var("NSW", 0, 2)
mc_solver.add(min_cons, max_cons)

V, min_cons, max_cons = create_var("V", 0, 2)
mc_solver.add(min_cons, max_cons)

T, min_cons, max_cons = create_var("T", 0, 2)
mc_solver.add(min_cons, max_cons)


# ### IMPLEMENTATION: Distinct Adjacent Colors Constraints
# As in the previous example, there are many valid ways to add constraints that enforce assigning different colors to adjacent regions of the map. One way is to add boolean constraints for each pair of adjacent regions, e.g., WA != SA; WA != NT; etc.
# 
# Another way is to use so-called pseudo-boolean cardinality constraint, which is a constraint of the form $ \sum w_i l_i = k $. Constraints of this form can be created in Z3 using `PbEq(((booleanA, w_A), (booleanB, w_B), ...), k)`. Distinct neighbors can be written with k=0, and w_i = 1 for all values of i. (Note: Z3 also has `PbLe()` for $\sum w_i l_i <= k $ and `PbGe()` for $\sum w_i l_i >= k $)
# 
# Choose one of the encodings discussed above and add the required constraints to the solver in the next cell.

# In[44]:


# TODO: add constraints to require adjacent regions to take distinct colorst
neighbors = [(WA, NT), (WA, SA), (NT, SA), (NT, Q), (SA, Q), (SA, NSW), (SA, V), (Q, NSW),
            (V, NSW)]
for n1, n2 in neighbors:
    mc_solver.add(n1!=n2)


# In[45]:


assert mc_solver.check() == sat, "Uh oh. The solver failed to find a solution. Check your constraints."

print("WA={}".format(colors[mc_solver.model()[WA].as_string()]))
print("NT={}".format(colors[mc_solver.model()[NT].as_string()]))
print("SA={}".format(colors[mc_solver.model()[SA].as_string()]))
print("Q={}".format(colors[mc_solver.model()[Q].as_string()]))
print("NSW={}".format(colors[mc_solver.model()[NSW].as_string()]))
print("V={}".format(colors[mc_solver.model()[V].as_string()]))
print("T={}".format(colors[mc_solver.model()[T].as_string()]))


# #### Map Coloring Challenge Problems
# 1. Generalize the procedure for this problem and try it on a larger map (countries in Africa, states in the USA, etc.)
# 2. Extend your procedure to perform [graph coloring](https://en.wikipedia.org/wiki/Graph_coloring) (maps are planar graphs; extending to all graphs generalizes the concept of "neighbors" to any pair of connected nodes). (Note: graph coloring is [NP-hard](https://en.wikipedia.org/wiki/Graph_coloring#Computational_complexity), so it may take a very long time to color large graphs.)

# ---
# ## III. N-Queens
# 
# In the next problem domain you'll solve the 8-queens puzzle, then use it to explore the complexity of solving CSPs. The 8-queens problem asks you to place 8 queens on a standard 8x8 chessboard such that none of the queens are in "check" (i.e., no two queens occupy the same row, column, or diagonal). The N-queens problem generalizes the puzzle to to any size square board.
# 
# ![The 8-queens problem is another classic CSP example](EightQueens.gif)
# 
# There are many acceptable ways to represent the N-queens problem, but one convenient way is to recognize that one of the constraints (either the row or column constraint) can be enforced implicitly by the encoding.  If we represent a solution as an array with N elements, then each position in the array can represent a column of the board, and the value at each position can represent which row the queen is placed on.
# 
# In this encoding, we only need a constraint to make sure that no two queens occupy the same row, and one to make sure that no two queens occupy the same diagonal.
# 
# #### IMPLEMENTATION: N-Queens Solver
# Complete the function below to take an integer N >= 5 and return a Z3 solver instance with appropriate constraints to solve the N-Queens problem. NOTE: it may take a few minutes for the solver to complete the suggested sizes below.

# In[18]:


def Abs(x):
    return If(x >= 0, x, -x)

def nqueens(N):
    nq_solver = Solver()
    # create variables which is the row number of the queens
    cols = [ Int('R_{}'.format(r + 1)) for r in range(N) ]
    # rows are different
    nq_solver.add(Distinct(cols))
    # rows are from 1 to N
    for r in cols:
        nq_solver.add(1<=r, r<=N)
    
    # diagonal contraints
    for r_ind in range(N):
        for other_r_ind in range(r_ind):
            if r_ind != other_r_ind:
                nq_solver.add(Abs(cols[r_ind]-cols[other_r_ind])!=0)
                
    return nq_solver 


# In[61]:


import time
from itertools import chain

runtimes = []
solutions = []
sizes = [8, 16, 32, 64]
run_time_size = {s: [] for s in sizes}

times = 1
for _ in range(5):
    for N in sizes:
        nq_solver = nqueens(N)
        start = time.perf_counter()
        assert nq_solver.check(), "Uh oh...The solver failed to find a solution. Check your constraints."
        end = time.perf_counter()
        print("{}-queens: {}ms".format(N, (end-start) * 1000))
        run_time_size[N].append((end-start)*1000)
        runtimes.append((end - start) * 1000)
        solutions.append(nq_solver)
        
        print(nq_solver.model())

# plt.plot(sizes, runtimes)
print(' variance {}'.format({s: np.var(run_time_size[s]) for s in sizes}))


# ### Queen Problem Challenges
# - Extend the loop to run several times and estimate the variance in the solver. How consistent is the solver timing between runs?
# - Read the `displayBoard()` function in the `util.py` module and use it to show your N-queens solution.

# ---
# ## IV. Revisiting Sudoku
# For the last CSP we'll revisit Sudoku from the first project. You previously solved Sudoku using backtracking search with constraint propagation. This time you'll re-write your solver using Z3. The backtracking search solver relied on domain-specific heuristics to select assignments during search, and to apply constraint propagation strategies (like elimination, only-choice, naked twins, etc.). The Z3 solver does not incorporate any domain-specific information, but makes up for that by incorporating a more sophisticated and a compiled solver routine.
# 
# ![Example of an easy sudoku puzzle](sudoku.png)

# In[ ]:


from itertools import chain  # flatten nested lists; chain(*[[a, b], [c, d], ...]) == [a, b, c, d, ...]
rows = 'ABCDEFGHI'
cols = '123456789'
boxes = [[Int("{}{}".format(r, c)) for c in cols] for r in rows]  # declare variables for each box in the puzzle
s_solver = Solver()  # create a solver instance


# #### IMPLEMENTATION: General Constraints
# Add constraints for each of the following conditions:
# - Boxes can only have values between 1-9 (inclusive)
# - Each box in a row must have a distinct value
# - Each box in a column must have a distinct value
# - Each box in a 3x3 block must have a distinct value

# In[ ]:

#
# # TODO: Add constraints that every box has a value between 1-9 (inclusive)
# s_solver.add( # YOUR CODE HERE )
#
# # TODO: Add constraints that every box in a row has a distinct value
# s_solver.add( # YOUR CODE HERE )
#
# # TODO: Add constraints that every box in a column has a distinct value
# s_solver.add( # YOUR CODE HERE )
#
# # TODO: Add constraints so that every box in a 3x3 block has a distinct value
# s_solver.add( # YOUR CODE HERE )


# #### IMPLMENTATION: Puzzle-Specific Constraints
# Given the hints provided in the initial puzzle layout, you must also add constraints binding the box values to the specified values. For example, to solve the example puzzle you must specify A3 == 3 and B1 == 9, etc. The cells with a value of zero in the board below are "blank", so you should **not** create any constraint with the associate box.

# In[ ]:


# use the value 0 to indicate that a box does not have an assigned value


# TODO: Add constraints boxes[i][j] == board[i][j] for each box where board[i][j] != 0
'''
board = ((0, 0, 3, 0, 2, 0, 6, 0, 0),
         (9, 0, 0, 3, 0, 5, 0, 0, 1),
         (0, 0, 1, 8, 0, 6, 4, 0, 0),
         (0, 0, 8, 1, 0, 2, 9, 0, 0),
         (7, 0, 0, 0, 0, 0, 0, 0, 8),
         (0, 0, 6, 7, 0, 8, 2, 0, 0),
         (0, 0, 2, 6, 0, 9, 5, 0, 0),
         (8, 0, 0, 2, 0, 3, 0, 0, 9),
         (0, 0, 5, 0, 1, 0, 3, 0, 0))
s_solver.add( # YOUR CODE HERE )


# In[ ]:


assert s_solver.check() == sat, "Uh oh. The solver didn't find a solution. Check your constraints."
for row, _boxes in enumerate(boxes):
    if row and row % 3 == 0:
        print('-'*9+"|"+'-'*9+"|"+'-'*9)
    for col, box in enumerate(_boxes):
        if col and col % 3 == 0:
            print('|', end='')
        print(' {} '.format(s_solver.model()[box]), end='')
    print()
'''


# #### Sudoku Challenges
# 1. Solve the "[hardest sudoku puzzle](# https://www.telegraph.co.uk/news/science/science-news/9359579/Worlds-hardest-sudoku-can-you-crack-it.html)"
# 2. Search for "3d Sudoku rules", then extend your solver to handle 3d puzzles

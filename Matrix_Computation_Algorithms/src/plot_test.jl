module Matrix_Computation_Algorithms
using Plots

# %%
var = 3 + 3

# %% 
var2 = var + 3

# %%
x = range(0, 10, length=100)
y = sin.(x)

plot(x, y)

end # module Matrix_Computation_Algorithms
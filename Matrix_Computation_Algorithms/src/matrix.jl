module Matrix_Computation_Algorithms
using LinearAlgebra

x = [1,2,3]
y = [4,5,6]

dot(x,y)
  
# %% time calculation of pi using MCA
function MCA_calc_pi(n)
    s = 0;
    for i = n:-1:1
        s += 1 / i^2
    end
    return sqrt(6*s)
  end;
n = 10^6
t = @elapsed p = MCA_calc_pi(n)
println("Approximation of pi with $n terms is $p, calculated in $t seconds.")

# %% matrix definition and operations
x = [1.0, 2.0, 3.0]
A = [11 12 13 14; 21 22 23 24; 31 32 33 34]

println("x = $x, A = $A")
println("x = ") ; display(x)
println("A = ") ; display(A)

display(diag(A))

# %% matrix transpose

At = A'
println("A' = ") ; display(At)

end
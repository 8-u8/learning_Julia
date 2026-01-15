module Matrix_Computation_Algorithms
using LinearAlgebra

# %% cholesky decomposition
function cholesky_decomposition(A::Matrix{Float64})
  n = size(A, 1)
  L = copy(LowerTriangular(float(A)))
  for j = 1:n
    L[j, j] = sqrt(L[j, j])
    L[j+1:n, j] = L[j+1:n, j] / L[j, j]
    for i = j+1:n
      L[i, j+1:i] -= L[i, j] * L[j+1:i, j]
    end
  end
  return L
end

# %% LDLt decomposition
function LDLt_decomposition(A::Matrix{Float64})
  n = size(A, 1)
  L = copy(LowerTriangular(float(A)))
  D = zeros(Float64, n)
  for j = 1:n
    D[j] = L[j, j]
    L[j, j] = 1.0
    L[j+1:n, j] = L[j+1:n, j] / D[j]
    for i = j+1:n
      L[i, j+1:i] -= D[j] * L[i, j] * L[j+1:i, j]
    end
  end
  return (L=L, D=D)
end

# %% example
Aspd = [4 2 3; 2 5 5.5; 3 5.5 8.5]
t = @elapsed L = cholesky_decomposition(Aspd)
println("Cholesky Decomposition L:")
display(L)
println("residual norm: ", norm(Aspd - L * L'))
println("Time elapsed for Cholesky Decomposition: ", t, " seconds")

F = LDLt_decomposition(Aspd)
println("LDLt Decomposition L:")
display(F.L)
println("LDLt Decomposition D:")
display(F.D)
println("residual norm: ", norm(Aspd - F.L * diagm(F.D) * F.L'))

end # module Matrix_Computation_Algorithms
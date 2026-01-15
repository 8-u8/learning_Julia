# %%
module Matrix_Computation_Algorithms
using LinearAlgebra

# %% define LU decomposition function

function LU_decomposition(A::Matrix{Float64})
    """
    LU分解… 正則な正方行列Aを、対角要素が1の下三角行列Lと上三角行列Uの積に分解する。
    入力: A - 正則な正方行列 (n x n)
    出力: L - 対角要素が1の下三角行列 (n x n)
          U - 上三角行列 (n x n)

    """
    # initialize L and U
    # memo: size(A)は行列の次元のタプルを返す。１番目の要素が行、２番目の要素が列
    n = size(A, 1) # number of rows
    L = zeros(n, n) # n x n zero matrix
    U = copy(float(A)) # make a copy of A to avoid modifying it

    # decomposition
    for j = 1:n-1
      L[j,j] = 1.0 # L
      for i = j+1:n
        l_ij = U[i, j] / U[j, j]
        U[i, j] = 0.0
        U[i, j+1:n] -= l_ij * U[j, j+1:n]
        L[i, j] = l_ij
      end
    end
    L[n, n] = 1.0

    return (L = L, U = U)
end
# %% LU decomposition with pivot function
function LU_decomposition_with_pivot(A::Matrix{Float64})
    """
    LU分解（ピボット選択付き）… 正則な正方行列Aでも、対角成分が0だと、LU_decompositionが失敗する場合がある。
    そこで、ピボット選択を行い、対角成分が0にならないようにする。
    入力: A - 正則な正方行列 (n x n)
    出力: L - 単位行列
          U - 上三角行列 (n x n)
          P - ピボット行列 (n x n)

    """
    # initialize L, U, and P
    n = size(A, 1) # number of rows
    L = Matrix{Float64}(I, n, n) # n x n identity matrix
    U = copy(float(A)) # make a copy of A to avoid modifying it
    p = Array(1:n)

    for j = 1:n-1
      # (partial) pivot selection
      pivot = argmax(abs.(U[j:n, j])) + (j - 1)
      p[j], p[pivot] = p[pivot], p[j] # swap p[j] and p[pivot]
      U[j, j:n], U[pivot, j:n] = U[pivot, j:n], U[j, j:n] # swap rows in U
      L[j, 1:j-1], L[pivot, 1:j-1] = L[pivot, 1:j-1], L[j, 1:j-1] # swap rows in L

      for i = j+1:n
        # ここはLU_decompositionと同じ
        l_ij = U[i, j] / U[j, j]
        U[i, j] = 0.0
        U[i, j+1:n] -= l_ij * U[j, j+1:n]
        L[i, j] = l_ij
      end
    end
    return (L=L, U=U, p=p)
end

# %% example
A = [1.0 2.0 3.0; 2.0 5.5 4.0; 0.5 4.0 6.5]
t = @elapsed F = LU_decomposition(A)

println("Matrix A:"); display(A)
println("Lower Triangular Matrix L:"); display(F.L)
println("Upper Triangular Matrix U:"); display(F.U)
println("residual norm: ", norm(A - F.L * F.U))
println("elapsed time: ", t, " seconds")

# %% example with pivot
A1 = [0 1; 1 0.1]
A2 = [1e-10 1; 1 0.1]
t1 = @elapsed F1 = LU_decomposition(A1)
t2 = @elapsed F2 = LU_decomposition(A2)

# 結果がめっちゃ変
F1
F2

t1_p = @elapsed F1_p = LU_decomposition_with_pivot(A1)
t2_p = @elapsed F2_p = LU_decomposition_with_pivot(A2)

println("Matrix A1:"); display(A1)
println("Lower Triangular Matrix L:"); display(F1_p.L)
println("Upper Triangular Matrix U:"); display(F1_p.U)
println("pivot:"); display(F1_p.p)
println("residual norm: ", norm(A1[F1_p.p,:] - F1_p.L * F1_p.U))
println("elapsed time: ", t1_p, " seconds")

println("Matrix A2:"); display(A2)
println("Lower Triangular Matrix L:"); display(F2_p.L)
println("Upper Triangular Matrix U:"); display(F2_p.U)
println("pivot:"); display(F2_p.p)
println("residual norm: ", norm(A2[F2_p.p,:] - F2_p.L * F2_p.U))
println("elapsed time: ", t2_p, " seconds")

# %%

end # module
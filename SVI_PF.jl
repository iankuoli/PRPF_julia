include("user_preference_train.jl");

#
# Test the performance tips
# Access arrays in memory order, along columns
#
A = sprand(5, 10, 0.1)
(i, j, v) = findnz(A)



A = sprand(170000, 100, 0.003)
u = rand(100, 100)
v = rand(17000, 10)
X = repmat(A, 100, 1)
Y = repmat(A, 1, 10)
@time broadcast!(*, X, X, u[:])
@time broadcast!(*, Y, Y, v[:]')

@time broadcast!(*, A, A, rand(170000, 1))

sum_tensorPhi = spzeros(Float64, 2000, 17000)


tensorPhi = spzeros(2000, 17000*100);
function WOW!(tensorPhi::SparseMatrixCSC{Float64,Int64}, A::SparseMatrixCSC{Float64,Int64}, u::Array{Float64,2}, v::Array{Float64,2}, K::Int64)
  for k = 1:K
    X = copy(A);
    Y = copy(A);
    broadcast!(*, X, X, u[:,k]);
    broadcast!(*, Y, Y, v[:,k]');
    tensorPhi[:, ((k-1)*17000+1):(k*17000)] = X + Y;
  end
end

@time computeTensorPhi!(tensorPhi, sum_tensorPhi, A, u, v, 17000, 10)
tensorPhi
sum_tensorPhi

@time for k = 1:10
  X = copy(A);
  Y = copy(A);
  print(string(k) * ", ");
  @time broadcast!(*, X, X, u[:,k]);
  @time broadcast!(*, Y, Y, v[:,k]');
  tensorPhi[:, ((k-1)*17000+1):(k*17000)] = X + Y;
end



#
# tensorPhi_{uik} = ρ_{uik} * x_{ui},
# where ρ_{uik} = ψ(Θ_{uk}) - log(Θ_{uk}) + ψ(β_{ik}) - log(β_{ik})
#
function identifyASparse(X::SparseMatrixCSC{Float64, Int64})
  (i_idx, j_idx) = findn(X);
  return sparse(i_idx, j_idx, ones(Float64, length(i_idx)), size(X,1), size(X,2))
end

function computeTensorPhi!{T}(tensorPhi::SparseMatrixCSC{Float64,Int64}, sum_tensorPhi::SparseMatrixCSC{Float64,Int64},
                              A::SparseMatrixCSC{Float64,Int64}, u::Array{Float64,2}, v::Array{Float64,2}, n::Int64, K::Int64)
  for k = 1:K
    X = copy(A);
    Y = copy(A);
    broadcast!(*, X, X, u[:,k]);
    broadcast!(*, Y, Y, v[:,k]');
    tensorPhi[:, ((k-1)*n+1):(k*n)] = X + Y;
    sum_tensorPhi[:,:] += tensorPhi[:, ((k-1)*n+1):(k*n)];
  end
  nothing
  #return tensorPhi, sum_tensorPhi
end

function setTensorPhi!{T}(tensorPhi::SparseMatrixCSC{Float64, Int64}, sum_tensorPhi::SparseMatrixCSC{Float64, Int64},
                          m::Int64, n::Int64 ,K::Int64,
                          matX_One::SparseMatrixCSC{Float64, Int64},
                          matTheta_Shp_psi::Array{Float64,2}, matTheta_Rte_log::Array{Float64,2},
                          matBeta_Shp_psi::Array{Float64,2}, matBeta_Rte_log::Array{Float64,2})
  print("k = ");
  tmpX = matTheta_Shp_psi - matTheta_Rte_log;
  tmpY = matBeta_Shp_psi - matBeta_Rte_log;

  computeTensorPhi!(tensorPhi, sum_tensorPhi, matX_One, tmpX, tmpY, n, K);

  nothing

  #for k = 1:K
  #  print(string(k) * ", ");
  #  X = copy(matX_One);
  #  Y = copy(matX_One);

  #  broadcast!(*, X, X, tmpX[:,k]);
  #  broadcast!(*, Y, Y, tmpY[:,k]');
  #  tensorPhi[:, ((k-1)*n+1):(k*n)] = X + Y;
  #  sum_tensorPhi += tensorPhi[:, ((k-1)*n+1):(k*n)];
  #end
  #return tensorPhi, sum_tensorPhi
end


function setTensorPhi2(m::Int64, n::Int64 ,K::Int64,
                      matX_One::SparseMatrixCSC{Float64, Int64}, tensorPhi::SparseMatrixCSC{Float64, Int64}, sum_tensorPhi::SparseMatrixCSC{Float64, Int64},
                      matTheta_Shp_psi::Array{Float64,2}, matTheta_Rte_log::Array{Float64,2},
                      matBeta_Shp_psi::Array{Float64,2}, matBeta_Rte_log::Array{Float64,2})
  @time tmpX = (matTheta_Shp_psi - matTheta_Rte_log)[:];
  @time tmpY = (matBeta_Shp_psi - matBeta_Rte_log)[:];
  @time X = repmat(matX_One, K, 1);
  @time Y = repmat(matX_One, 1, K);
  @time broadcast!(*, X, X, tmpX);
  @time broadcast!(*, Y, Y, tmpY');

  for k = 1:K
    tensorPhi[:, ((k-1)*n+1):(k*n)] = X[((k-1)*m+1):(k*m), :] + Y[:, ((k-1)*n+1):(k*n)];
    sum_tensorPhi += tensorPhi[:, ((k-1)*n+1):(k*n)];
  end
  nothing
  #return tensorPhi, sum_tensorPhi
end


function Update_tensorPhi!{T}(tensorPhi::SparseMatrixCSC{Float64, Int64},
                              predict_X::SparseMatrixCSC{Float64,Int64},
                              matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                              matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2})

  m = size(predict_X, 1);
  n = size(predict_X, 2);
  K = size(matTheta_Shp, 2);

  tensorPhi = spzeros(m, n*K);
  sum_tensorPhi = spzeros(m, n);

  matTheta_Shp_psi = digamma(matTheta_Shp);
  matTheta_Rte_log = log(matTheta_Rte);
  matBeta_Shp_psi = digamma(matBeta_Shp);
  matBeta_Rte_log = log(matBeta_Rte);

  matX_One = identifyASparse(predict_X);

  @time setTensorPhi!(tensorPhi, sum_tensorPhi, m, n, K, matX_One, matTheta_Shp_psi, matTheta_Rte_log, matBeta_Shp_psi, matBeta_Rte_log);

  map!((x) -> 1 ./ x, nonzeros(sum_tensorPhi));

  for k = 1:K
    tensorPhi[:, ((k-1)*n+1):(k*n)] = predict_X .* tensorPhi[:, ((k-1)*n+1):(k*n)] .* sum_tensorPhi;
  end

  nothing
  #return tensorPhi
end


function Update_matTheta!{T}(matTheta::Array{Float64,2}, matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                             M::Int64, N::Int64, K::Int64, usr_batch_size::Int64,
                             lr::Float64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                             matX_train::SparseMatrixCSC{Float64,Int64}, tensorPhi::SparseMatrixCSC{Float64,Int64},
                             matBeta::Array{Float64,2}, a::Float64, matEpsilon::Array{Float64,1})

  m = length(usr_idx);
  n = length(itm_idx);

  for k = 1:K
    matTheta_Shp[usr_idx, k] = (1 - lr) * matTheta_Shp[usr_idx, k] + lr * (a + sum(tensorPhi[:, ((k-1)*n+1):(k*n)], 2));
  end
  matTheta_Rte[usr_idx,:] = (1 - lr) * matTheta_Rte[usr_idx, :] + lr * broadcast(+, repmat(sum(matBeta[itm_idx,:], 1), m, 1), matEpsilon[usr_idx]);
  matTheta[usr_idx,:] = matTheta_Shp[usr_idx,:] ./ matTheta_Rte[usr_idx,:];

  nothing
  #return matTheta, matTheta_Shp, matTheta_Rte
end


function Update_matBeta!{T}(matBeta::Array{Float64,2}, matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2},
                            M::Int64, N::Int64, K::Int64, usr_batch_size::Int64,
                            lr::Float64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                            matX_train::SparseMatrixCSC{Float64,Int64}, tensorPhi::SparseMatrixCSC{Float64,Int64},
                            matTheta::Array{Float64,2}, d::Float64, matEta::Array{Float64,1})

  m = length(usr_idx);
  n = length(itm_idx);

  if usr_batch_size == M
      scale = ones(length(itm_idx), 1);
  else
      scale = sum(matX_train[:, itm_idx] .> 0, 1)' ./ sum(matX_train[usr_idx, itm_idx] .> 0, 1)';
  end

  for k = 1:K
    matBeta_Shp[itm_idx, k] = (1 - lr) * matBeta_Shp[itm_idx, k] + lr * (d + scale .* sum(tensorPhi[:, ((k-1)*n+1):(k*n)], 1)');
  end
  matBeta_Rte[itm_idx,:] = (1 - lr) * matBeta_Rte[itm_idx, :] + lr * broadcast(+, scale * sum(matTheta[usr_idx,:], 1), matEta[itm_idx]);
  matBeta[itm_idx,:] = matBeta_Shp[itm_idx,:] ./ matBeta_Rte[itm_idx,:];

  nothing
  #return matBeta, matBeta_Shp, matBeta_Rte
end


function Update_matEpsilon!{T}(matEpsilon::Array{Float64,1}, matEpsilon_Shp::Array{Float64,1}, matEpsilon_Rte::Array{Float64,1},
                               lr::Float64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                               matTheta::Array{Float64,2}, a::Float64, b::Float64, c::Float64)
  K = size(matTheta, 2);
  matEpsilon_Shp[usr_idx] = (1-lr) * matEpsilon_Shp[usr_idx] + lr * (b + K * a);
  matEpsilon_Rte[usr_idx] = (1-lr) * matEpsilon_Rte[usr_idx] + lr * (c + sum(matTheta[usr_idx,:], 2)[:]);
  matEpsilon[usr_idx] = matEpsilon_Shp[usr_idx] ./ matEpsilon_Rte[usr_idx];
  nothing
  #return matEpsilon, matEpsilon_Shp, matEpsilon_Rte
end


function Update_matEta!{T}(matEta::Array{Float64,1}, matEta_Shp::Array{Float64,1}, matEta_Rte::Array{Float64,1},
                           lr::Float64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                           matBeta::Array{Float64,2}, d::Float64, e::Float64, f::Float64)
  K = size(matBeta, 2);
  matEta_Shp[itm_idx] = (1-lr) * matEta_Shp[itm_idx] + lr * (e + K * d);
  matEta_Rte[itm_idx] = (1-lr) * matEta_Rte[itm_idx] + lr * (f + sum(matBeta[itm_idx,:], 2)[:]);
  matEta[itm_idx] = matEta_Shp[itm_idx] ./ matEta_Rte[itm_idx];
  nothing
  #return matEta, matEta_Shp, matEta_Rte
end


function SVI_PF(lr::Float64, M::Int64, N::Int64, K::Int64, usr_batch_size::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                predict_X::SparseMatrixCSC{Float64,Int64}, matX_train::SparseMatrixCSC{Float64,Int64},
                matTheta::Array{Float64,2}, matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                matBeta::Array{Float64,2}, matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2},
                matEpsilon::Array{Float64,1}, matEpsilon_Shp::Array{Float64,1}, matEpsilon_Rte::Array{Float64,1},
                matEta::Array{Float64,1}, matEta_Shp::Array{Float64,1}, matEta_Rte::Array{Float64,1},
                prior::Tuple{Float64,Float64,Float64,Float64,Float64,Float64})

  m = length(usr_idx);
  n = length(itm_idx);
  (a,b,c,d,e,f) = prior;

  #
  # Update tensorPhi
  #
  println("Update tensorPhi ... ");
  tensorPhi = spzeros(m, n*K);
  Update_tensorPhi!(tensorPhi, predict_X, matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:]);

  #
  # Update latent variables
  #
  println("Update latent variables ... ");
  Update_matTheta(M, N, K, usr_batch_size, lr, usr_idx, itm_idx, matX_train, tensorPhi, matBeta, a, matEpsilon, matTheta, matTheta_Shp, matTheta_Rte);

  Update_matBeta(M, N, K, usr_batch_size, lr, usr_idx, itm_idx, matX_train, tensorPhi, matTheta, d, matEta, matBeta, matBeta_Shp, matBeta_Rte);

  Update_matEpsilon(lr, usr_idx, itm_idx, matTheta, a, b, c, matEpsilon, matEpsilon_Shp, matEpsilon_Rte);

  Update_matEta(lr, usr_idx, itm_idx, matBeta, d, e, f, matEta, matEta_Shp, matEta_Rte);
  nothing
end


#
#  /// --- Unit test for function: Update_tensorPhi(), Update_matTheta(), Update_matBeta() --- ///
#
# predict_X =  sparse([5. 4 3 0 0 0 0 0;
#                      3. 4 5 0 0 0 0 0;
#                      0  0 0 3 3 4 0 0;
#                      0  0 0 5 4 5 0 0;
#                      0  0 0 0 0 0 5 4;
#                      0  0 0 0 0 0 3 4;
#                      0  0 0 0 0 0 0 0])
# matTheta_Shp = [1. 0 0; 1 0 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 0]+10e-10
# matTheta_Rte = [1. 0 0; 1 0 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 0]+10e-10
# matBeta_Shp = [4. 0 0; 4 0 0; 4 0 0; 0 4 0; 0 3 0; 0 5 0; 0 0 4; 0 0 4]+10e-10
# matBeta_Rte = [4. 0 0; 4 0 0; 4 0 0; 0 4 0; 0 3 0; 0 5 0; 0 0 4; 0 0 4]+10e-10
# lr=1.
# matTheta = matTheta_Shp ./ matTheta_Rte
# matBeta = matBeta_Shp ./ matBeta
# M = 7
# N = 8
# usr_batch_size = 7
# usr_idx = [1,2,3,4,5,6]
# itm_idx = [1,2,3,4,5,6,7,8]
# a = 0.3
# b = 0.3
# c = 0.3
# d = 0.3
# e = 0.3
# f = 0.3
# matEpsilon = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0]
# matEpsilon_Shp = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0]
# matEpsilon_Rte = [1., 1., 1., 1., 1., 1., 0]
# matEta = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
# matEta_Shp = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
# matEta_Rte = [1., 1., 1., 1., 1., 1., 1., 1.]
# tensorPhi = Update_tensorPhi(predict_X[usr_idx,itm_idx], matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:])
# matTheta, matTheta_Shp, matTheta_Rte = Update_matTheta(M, N, K, usr_batch_size, lr, usr_idx, itm_idx,
#                                                        predict_X, tensorPhi, matBeta, a, matEpsilon,
#                                                        matTheta, matTheta_Shp, matTheta_Rte)
#
#
# matBeta, matBeta_Shp, matBeta_Rte = Update_matBeta(M, N, K, usr_batch_size, lr, usr_idx, itm_idx,
#                                                    predict_X, tensorPhi, matTheta, d, matEta,
#                                                    matBeta, matBeta_Shp, matBeta_Rte)
#
#
# matEpsilon, matEpsilon_Shp, matEpsilon_Rte = Update_matEpsilon(lr, usr_idx, itm_idx, matTheta, a, b, c, matEpsilon, matEpsilon_Shp, matEpsilon_Rte)
#
#
# matEta, matEta_Shp, matEta_Rte = Update_matEta(lr, usr_idx, itm_idx, matBeta, d, e, f, matEta, matEta_Shp, matEta_Rte)

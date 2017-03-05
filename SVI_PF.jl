include("user_preference_train.jl");

#
# tensorPhi_{uik} = ρ_{uik} * x_{ui},
# where ρ_{uik} = ψ(Θ_{uk}) - log(Θ_{uk}) + ψ(β_{ik}) - log(β_{ik})
#
function Update_tensorPhi(predict_X::SparseMatrixCSC{Float64,Int64},
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

  matX_One = predict_X .> 0;
  for k = 1:K
    tensorPhi[:, ((k-1)*n+1):(k*n)] = broadcast(*, matX_One, (matTheta_Shp_psi[:,k] - matTheta_Rte_log[:,k])) +
                                      broadcast(*, matX_One, (matBeta_Shp_psi[:,k] - matBeta_Rte_log[:,k])');
    sum_tensorPhi += tensorPhi[:, ((k-1)*n+1):(k*n)];
  end

  map!((x) -> 1 ./ x, nonzeros(sum_tensorPhi))

  for k = 1:K
    tensorPhi[:, ((k-1)*n+1):(k*n)] = predict_X .* tensorPhi[:, ((k-1)*n+1):(k*n)] .* sum_tensorPhi;
  end

  return tensorPhi
end


function Update_matTheta(M::Int64, N::Int64, K::Int64, usr_batch_size::Int64,
                         lr::Float64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                         predict_X::SparseMatrixCSC{Float64,Int64}, tensorPhi::SparseMatrixCSC{Float64,Int64},
                         matBeta::Array{Float64,2}, a::Float64, matEpsilon::Array{Float64,1},
                         matTheta::Array{Float64,2}, matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2})

  m = length(usr_idx);
  n = length(itm_idx);

  for k = 1:K
    matTheta_Shp[usr_idx, k] = (1 - lr) * matTheta_Shp[usr_idx, k] + lr * (a + sum(tensorPhi[:, ((k-1)*n+1):(k*n)], 2));
  end
  matTheta_Rte[usr_idx,:] = (1 - lr) * matTheta_Rte[usr_idx, :] + lr * broadcast(+, repmat(sum(matBeta[itm_idx,:], 1), m, 1), matEpsilon[usr_idx,:]);
  matTheta[usr_idx,:] = matTheta_Shp[usr_idx,:] ./ matTheta_Rte[usr_idx,:];

  return matTheta, matTheta_Shp, matTheta_Rte
end


function Update_matBeta(M::Int64, N::Int64, K::Int64, usr_batch_size::Int64,
                        lr::Float64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                        predict_X::SparseMatrixCSC{Float64,Int64}, tensorPhi::SparseMatrixCSC{Float64,Int64},
                        matTheta::Array{Float64,2}, d::Float64, matEta::Array{Float64,1},
                        matBeta::Array{Float64,2}, matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2})

  m = length(usr_idx);
  n = length(itm_idx);

  if usr_batch_size == M
      scale = ones(length(itm_idx), 1);
  else
      scale = sum(predict_X[:, itm_idx] .> 0, 1)' ./ sum(predict_X[usr_idx, itm_idx] .> 0, 1)';
  end

  for k = 1:K
    matBeta_Shp[itm_idx, k] = (1 - lr) * matBeta_Shp[itm_idx, k] + lr * (a + scale .* sum(tensorPhi[:, ((k-1)*n+1):(k*n)], 1)');
  end
  matBeta_Rte[itm_idx,:] = (1 - lr) * matBeta_Rte[itm_idx, :] + lr * broadcast(+, scale * sum(matTheta[usr_idx,:], 1), matEta[itm_idx,:]);
  matBeta[itm_idx,:] = matBeta_Shp[itm_idx,:] ./ matBeta_Rte[itm_idx,:];

  return matBeta, matBeta_Shp, matBeta_Rte
end


function Update_matEpsilon(lr::Float64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                           matTheta::Array{Float64,2}, a::Float64, b::Float64, c::Float64,
                           matEpsilon::Array{Float64,1}, matEpsilon_Shp::Array{Float64,1}, matEpsilon_Rte::Array{Float64,1})
  K = size(matTheta, 2);
  matEpsilon_Shp[usr_idx] = (1-lr) * matEpsilon_Shp[usr_idx] + lr * (b + K * a);
  matEpsilon_Rte[usr_idx] = (1-lr) * matEpsilon_Rte[usr_idx] + lr * (c + sum(matTheta[usr_idx,:], 2));
  matEpsilon[usr_idx] = matEpsilon_Shp[usr_idx] ./ matEpsilon_Rte[usr_idx];
  return matEpsilon, matEpsilon_Shp, matEpsilon_Rte
end


function Update_matEta(lr::Float64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                       matBeta::Array{Float64,2}, d::Float64, e::Float64, f::Float64,
                       matEta::Array{Float64,1}, matEta_Shp::Array{Float64,1}, matEta_Rte::Array{Float64,1})
  K = size(matBeta, 2);
  matEta_Shp[itm_idx] = (1-lr) * matEta_Shp[itm_idx] + lr * (e + K * d);
  matEta_Rte[itm_idx] = (1-lr) * matEta_Rte[itm_idx] + lr * (f + sum(matBeta[itm_idx,:], 2));
  matEta[itm_idx] = matEta_Shp[itm_idx] ./ matEta_Rte[itm_idx];
  return matEta, matEta_Shp, matEta_Rte
end


function SVI_PF(lr::Float64, M::Int64, N::Int64, K::Int64, ini_scale::Float64, usr_batch_size::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                predict_X::SparseMatrixCSC{Float64,Int64},
                matTheta::Array{Float64,2}, matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                matBeta::Array{Float64,2}, matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2},
                matEpsilon::Array{Float64,2}, matEpsilon_Shp::Array{Float64,2}, matEpsilon_Rte::Array{Float64,2},
                matEta::Array{Float64,2}, matEta_Shp::Array{Float64,2}, matEta_Rte::Array{Float64,2},
                prior::Vector{Float64})

  m = length(usr_idx);
  n = length(itm_idx);
  (a,b,c,d,e,f) = prior;

  #
  # Update tensorPhi
  #
  print("Update tensorPhi ... ");
  tensorPhi = Update_tensorPhi(predict_X[usr_idx,itm_idx], matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:]);

  #
  # Update latent variables
  #
  matTheta, matTheta_Shp, matTheta_Rte = Update_matTheta(M, N, K, usr_batch_size, lr, usr_idx, itm_idx,
                                                         predict_X, tensorPhi, matBeta, a, matEpsilon,
                                                         matTheta, matTheta_Shp, matTheta_Rte);

  matBeta, matBeta_Shp, matBeta_Rte = Update_matBeta(M, N, K, usr_batch_size, lr, usr_idx, itm_idx,
                                                     predict_X, tensorPhi, matTheta, d, matEta,
                                                     matBeta, matBeta_Shp, matBeta_Rte);

  matEpsilon, matEpsilon_Shp, matEpsilon_Rte = Update_matEpsilon(lr, usr_idx, itm_idx, matTheta, a, b, c,
                                                                 matEpsilon, matEpsilon_Shp, matEpsilon_Rte);

  matEta, matEta_Shp, matEta_Rte = Update_matEta(lr, usr_idx, itm_idx, matBeta, d, e, f,
                                                 matEta, matEta_Shp, matEta_Rte);
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

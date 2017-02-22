include("user_preference_train.jl");

#
# tensorPhi_{uik} = ρ_{uik} * x_{ui},
# where ρ_{uik} = ψ(Θ_{uk}) - log(Θ_{uk}) + ψ(β_{ik}) - log(β_{ik})
#
function Update_tensorPhi(predict_X::SparseMatrixCSC{Float64,2},
                          matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                          matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2})

  tensorPhi = spzeros(m, n*K);
  sum_tensorPhi = spzeros(m, n);

  matTheta_Shp_psi = psi(matTheta_Shp);
  matTheta_Rte_log = log(matTheta_Rte);
  matBeta_Shp_psi = psi(matBeta_Shp);
  matBeta_Rte_log = log(matBeta_Rte);

  matX_One = predict_X > 0;
  for k = 1:K
    tensorPhi[m, ((k-1)*n+1):(k*n)] = broadcast(*, matX_One, (matTheta_Shp_psi[:,k] - matTheta_Rte_log[:,k])) +
                                      broadcast(*, matX_One, (matBeta_Shp_psi[:,k] - matBeta_Rte_log[:,k])');
    sum_tensorPhi += tensorPhi[m, ((k-1)*n+1):(k*n)];
  end
  for k = 1:K
    tensorPhi[m, ((k-1)*n+1):(k*n)] = predict_X .* tensorPhi[m, ((k-1)*n+1):(k*n)] ./ sum_tensorPhi;
  end

  return tensorPhi
end


function Update_matTheta(lr::Float64, tensorPhi::SparseMatrixCSC{Float64,2},
                         matBeta::Array{Float64,2}, a::Float64, matEpsilon::Array{Float64,1},
                         matTheta::Array{Float64,2}, matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2})
  m = size(matTheta, 1);
  n = size(matBeta, 1);
  K = size(matTheta, 2);

  for k = 1:K
    matTheta_Shp[:, k] = (1 - lr) * matTheta_Shp[:, k] + lr * (a + sum(tensorPhi[:, ((k-1)*n+1):(k*n)], 2));
  end
  matTheta_Rte = (1 - lr) * matTheta_Rte[:, k] + lr * broadcast(+, repmat(sum(matBeta, 1) m, 1), matEpsilon);
  matTheta = matTheta_Shp ./ matTheta_Rte;

  return matTheta, matTheta_Shp, matTheta_Rte
end


function Update_matBeta(lr::Float64, tensorPhi::SparseMatrixCSC{Float64,2},
                        matTheta::Array{Float64,2}, d::Float64, matEta::Array{Float64,1},
                        matBeta::Array{Float64,2}, matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2})
  m = size(matTheta, 1);
  n = size(matBeta, 1);
  K = size(matTheta, 2);

  for k = 1:K
    matBeta_Shp[:, k] = (1 - lr) * matBeta_Shp[:, k] + lr * (a + sum(tensorPhi[:, ((k-1)*n+1):(k*n)], 1)');
  end
  matBeta_Rte = (1 - lr) * matBeta_Rte[:, k] + lr * broadcast(+, repmat(sum(matTheta, 1) n, 1), matEta);
  matBeta = matBeta_Shp ./ matBeta_Rte;

  return matBeta, matBeta_Shp, matBeta_Rte
end


function Update_matEpsilon(lr::Float64, matTheta::Array{Float64,2}, a::Float64, b::Float64, c::Float64,
                           matEpsilon::Array{Float64,1}, matEpsilon_Shp::Array{Float64,1}, matEpsilon_Rte::Array{Float64,1})
  K = size(matTheta, 2);
  matEpsilon_Shp = (1-lr) * matEpsilon_Shp + lr * (b + K * a);
  matEpsilon_Rte = (1-lr) * matEpsilon_Rte + lr * (c + sum(matTheta, 2));
  matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte;
  return matEpsilon, matEpsilon_Shp, matEpsilon_Rte
end


function Update_matEta(lr::Float64, matBeta::Array{Float64,2}, d::Float64, e::Float64, f::Float64,
                       matEta::Array{Float64,1}, matEta_Shp::Array{Float64,1}, matEta_Rte::Array{Float64,1})
  K = size(matBeta, 2);
  matEta_Shp = (1-lr) * matEta_Shp + lr * (e + K * d);
  matEta_Rte = (1-lr) * matEta_Rte + lr * (f + sum(matBeta, 2));
  matEta = matEta_Shp ./ matEta_Rte;
  return matEta, matEta_Shp, matEta_Rte
end


function user_preference_train(lr::Float64, M::UInt64, N::UInt64, K::UInt16, ini_scale::Float64, usr_idx::Array{Int64,1}, itm_idx::Array::Array{Int64,1},
                               predict_X::SparseMatrixCSC{Float64,2},
                               matTheta::Array{Float64,2}, matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                               matBeta::Array{Float64,2}, matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2},
                               matEpsilon::Array{Float64,1}, matEpsilon_Shp::Array{Float64,1}, matEpsilon_Rte::Array{Float64,1},
                               matEta::Array{Float64,1}, matEta_Shp::Array{Float64,1}, matEta_Rte::Array{Float64,1})

  m = legnth(usr_idx);
  n = legnth(itm_idx);

  #
  # Update tensorPhi
  #
  print("Update tensorPhi ... ");
  tensorPhi = Update_tensorPhi(predict_X[usr_idx, itm_idx], matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:]);

  matTheta[usr_idx,:], matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:] = Update_matTheta(lr, tensorPhi, matBeta[itm_idx,:], a, matEpsilon[usr_idx],
                                                                                          matTheta[usr_idx,:], matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:]);

  matBeta[itm_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:] = Update_matBeta(lr, tensorPhi, matTheta[usr_idx,:], d, matEta[itm_idx],
                                                                                      matBeta[itm_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:]);

  matEpsilon[usr_idx], matEpsilon_Shp[usr_idx], matEpsilon_Rte[usr_idx] = Update_matEpsilon(lr, matTheta[usr_idx,:], a, b, c,
                                                                                            matEpsilon[usr_idx], matEpsilon_Shp[usr_idx], matEpsilon_Rte[usr_idx]);

  matEta[usr_idx], matEta_Shp[usr_idx], matEta_Rte[usr_idx] = Update_matEta(lr, matBeta[itm_idx,:], d, e, f,
                                                                            matEta[itm_idx], matEta_Shp[itm_idx], matEta_Rte[itm_idx]);
end

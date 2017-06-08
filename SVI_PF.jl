include("user_preference_train.jl");

#
# Test the performance tips
# testing for computationality complexity with large scale
#
#m = 2000
#n = 17000
#K = 100
#A = sprand(m, n, 0.01)
#tensorPhi = spzeros(m, n*K)
#sum_tensorPhi = spzeros(m, n)
#matTheta_Shp = rand(m, K)
#matTheta_Rte = rand(m, K)
#matBeta_Shp = rand(n, K)
#matBeta_Rte = rand(n, K)
#@time tensorPhi = Update_tensorPhi(A, matTheta_Shp, matTheta_Rte, matBeta_Shp, matBeta_Rte)

#
# Testing for value accuracy
#
#A = sparse([3. 0 0 8; 4 5 0 0; 0 0 6 8; 0 8 6 0])
#matTheta_Shp = [10. 1; 1 8; 8 1; 1 10]
#matTheta_Rte = [2. 2; 2 2; 3 3; 2 2]
#matBeta_Shp = [10. 1; 1 8; 8 1; 1 10]
#matBeta_Rte = [2. 2; 2 2; 3 3; 2 2]
#@time tensorPhi = Update_tensorPhi(A, matTheta_Shp, matTheta_Rte, matBeta_Shp, matBeta_Rte)
#xx = full(tensorPhi)
#full(tensorPhi[:,1:4] + tensorPhi[:,5:8])


function Update_tensorPhi(predict_X::SparseMatrixCSC{Float64,Int64},
                          matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                          matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2})

  m = size(predict_X, 1)
  n = size(predict_X, 2)
  K = size(matTheta_Shp, 2)

  tensorPhi = spzeros(m, n*K)
  sum_tensorPhi = spzeros(m, n)

  tmpX = digamma(matTheta_Shp) - log(matTheta_Rte)
  tmpY = digamma(matBeta_Shp) - log(matBeta_Rte)

  (i, j, v) = findnz(predict_X)
  nnz_X = length(i)
  is = zeros(Int64, nnz_X * K)
  js = zeros(Int64, nnz_X * K)
  vs = zeros(Float64, nnz_X * K)
  @time for k=1:K
    is[((k-1)*nnz_X+1):(k*nnz_X)] = i
    js[((k-1)*nnz_X+1):(k*nnz_X)] = j + (k-1)*n
    vs[((k-1)*nnz_X+1):(k*nnz_X)] = tmpX[i,k] + tmpY[j,k]
  end

  tensorPhi = sparse(is, js, exp(vs), m, n*K)

  for k=1:K
    sum_tensorPhi += tensorPhi[:,((k-1)*n+1):(k*n)];
  end

  (i_sum, j_sum, v_sum) = findnz(sum_tensorPhi);
  (i_phi, j_phi, v_phi) = findnz(tensorPhi);
  v_norm = ones(Float64, nnz_X * K);

  for k = 1:K
    v_norm[((k-1)*nnz_X+1):(k*nnz_X)] = v .* v_phi[((k-1)*nnz_X+1):(k*nnz_X)] ./ v_sum;
  end

  retNormalizedPhi = sparse(i_phi, j_phi, v_norm);

  return retNormalizedPhi
end


function Update_matTheta(matTheta::Array{Float64,2}, matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                         usr_batch_size::Int64, lr::Float64,
                         tensorPhi::SparseMatrixCSC{Float64,Int64},
                         matBeta::Array{Float64,2}, a::Float64, matEpsilon::Array{Float64,1})
  K = size(matTheta, 2)
  m = size(tensorPhi,1)
  n = Int(size(tensorPhi,2) / K)

  for k = 1:K
    matTheta_Shp[:, k] = (1 - lr) * matTheta_Shp[:, k] + lr * (a + sum(tensorPhi[:, ((k-1)*n+1):(k*n)], 2))
  end
  matTheta_Rte = (1 - lr) * matTheta_Rte + lr * broadcast(+, repmat(sum(matBeta, 1), m, 1), matEpsilon)
  matTheta = matTheta_Shp ./ matTheta_Rte

  return matTheta, matTheta_Shp, matTheta_Rte
end


function Update_matBeta(matBeta::Array{Float64,2}, matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2},
                        usr_batch_size::Int64, lr::Float64,
                        tensorPhi::SparseMatrixCSC{Float64,Int64},
                        matTheta::Array{Float64,2}, d::Float64, matEta::Array{Float64,1}, scale::Array{Float64,1}, )
  K = size(matBeta, 2)
  m = size(tensorPhi,1)
  n = Int(size(tensorPhi,2) / K)

  for k = 1:K
    matBeta_Shp[:, k] = (1 - lr) * matBeta_Shp[:, k] + lr * (d + scale .* sum(tensorPhi[:, ((k-1)*n+1):(k*n)], 1)')
  end
  matBeta_Rte = (1 - lr) * matBeta_Rte + lr * broadcast(+, scale * sum(matTheta, 1), matEta)
  matBeta = matBeta_Shp ./ matBeta_Rte

  #nothing
  return matBeta, matBeta_Shp, matBeta_Rte
end


function Update_matEpsilon(matEpsilon::Array{Float64,1}, matEpsilon_Shp::Array{Float64,1}, matEpsilon_Rte::Array{Float64,1},
                           lr::Float64, matTheta::Array{Float64,2}, a::Float64, b::Float64, c::Float64)
  K = size(matTheta, 2);

  matEpsilon_Shp = (1-lr) * matEpsilon_Shp + lr * (b + K * a);
  matEpsilon_Rte = (1-lr) * matEpsilon_Rte + lr * (c + sum(matTheta, 2)[:]);
  matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte;

  return matEpsilon, matEpsilon_Shp, matEpsilon_Rte
end


function Update_matEta(matEta::Array{Float64,1}, matEta_Shp::Array{Float64,1}, matEta_Rte::Array{Float64,1},
                       lr::Float64, matBeta::Array{Float64,2}, d::Float64, e::Float64, f::Float64)
  K = size(matBeta, 2);

  matEta_Shp = (1-lr) * matEta_Shp + lr * (e + K * d);
  matEta_Rte = (1-lr) * matEta_Rte + lr * (f + sum(matBeta, 2)[:]);
  matEta = matEta_Shp ./ matEta_Rte;

  return matEta, matEta_Shp, matEta_Rte
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
  tensorPhi = Update_tensorPhi(predict_X, matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:]);

  #
  # Update latent variables
  #
  println("Update latent variables ... ");

  #  ---- Update matTheta ---  #
  matTheta[usr_idx,:],
  matTheta_Shp[usr_idx,:],
  matTheta_Rte[usr_idx,:] = Update_matTheta(matTheta[usr_idx,:], matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:],
                                            usr_batch_size, lr, tensorPhi, matBeta[itm_idx,:], a, matEpsilon[usr_idx]);

  #  ---- Update matBeta ---  #
  if usr_batch_size == M
      scale = ones(length(itm_idx), 1)
  else
      scale = sum(matX_train[:, itm_idx] .> 0, 1)' ./ sum(matX_train[usr_idx, itm_idx] .> 0, 1)'
  end
  matBeta[itm_idx,:],
  matBeta_Shp[itm_idx,:],
  matBeta_Rte[itm_idx,:] = Update_matBeta(matBeta[itm_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:],
                                          usr_batch_size, lr, tensorPhi, matTheta[usr_idx,:], d, matEta[itm_idx], scale[:]);


  #  ---- Update matEpsilon ---  #
  matEpsilon[usr_idx],
  matEpsilon_Shp[usr_idx],
  matEpsilon_Rte[usr_idx] = Update_matEpsilon(matEpsilon[usr_idx], matEpsilon_Shp[usr_idx], matEpsilon_Rte[usr_idx],
                                              lr, matTheta[usr_idx,:], a, b, c);


  #  ---- Update matEta ---  #
  matEta[itm_idx],
  matEta_Shp[itm_idx],
  matEta_Rte[itm_idx] = Update_matEta(matEta[itm_idx], matEta_Shp[itm_idx], matEta_Rte[itm_idx],
                                      lr, matBeta[itm_idx,:], d, e, f);

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

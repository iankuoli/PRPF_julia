include("user_preference_train.jl");


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


function SVI_PF2(lr::Float64, M::Int64, N::Int64, K::Int64, usr_batch_size::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                 predict_X::SparseMatrixCSC{Float64,Int64}, matX_train::SparseMatrixCSC{Float64,Int64},
                 matTheta::Array{Float64,2}, matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                 matBeta::Array{Float64,2}, matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2},
                 matEpsilon::Array{Float64,1}, matEpsilon_Shp::Array{Float64,1}, matEpsilon_Rte::Array{Float64,1},
                 matEta::Array{Float64,1}, matEta_Shp::Array{Float64,1}, matEta_Rte::Array{Float64,1},
                 prior::Tuple{Float64,Float64,Float64,Float64,Float64,Float64})

  m = length(usr_idx);
  n = length(itm_idx);
  K = size(matTheta_Shp, 2)
  (a,b,c,d,e,f) = prior;

  #
  # Update tensorPhi
  #
  println("Update (is, js, vs) of tensorPhi ... ");

  tmpX = digamma(matTheta_Shp[usr_idx,:]) - log(matTheta_Rte[usr_idx,:])
  tmpY = digamma(matBeta_Shp[itm_idx,:]) - log(matBeta_Rte[itm_idx,:])

  (i, j, v) = findnz(predict_X)
  nnz_X = length(i)
  vs = zeros(Float64, nnz_X * K)
  vs_sum = zeros(Float64, nnz_X)
  @time for k=1:K
    vs[((k-1)*nnz_X+1):(k*nnz_X)] = exp(tmpX[i,k] + tmpY[j,k])
    vs_sum = vs_sum + vs[((k-1)*nnz_X+1):(k*nnz_X)]
  end
  @time for k=1:K
    vs[((k-1)*nnz_X+1):(k*nnz_X)] ./= vs_sum
  end
  vs_sum = 0


  #
  # Update latent variables
  #
  println("Update latent variables ... ");


  #  ---- Update matTheta & matBeta ---  #
  if usr_batch_size == M
      scale = ones(length(itm_idx), 1)
  else
      scale = sum(matX_train[:, itm_idx] .> 0, 1)' ./ sum(matX_train[usr_idx, itm_idx] .> 0, 1)'
  end

  for k = 1:K
    tensorPhi = sparse(i, j, vs[((k-1)*nnz_X+1):(k*nnz_X)], m, n)
    matTheta_Shp[usr_idx, k] = (1 - lr) * matTheta_Shp[usr_idx, k] + lr * (a + sum(tensorPhi, 2))
    matBeta_Shp[itm_idx, k] = (1 - lr) * matBeta_Shp[itm_idx, k] + lr * (d + scale .* sum(tensorPhi, 1)')
  end

  matTheta_Rte[usr_idx,:] = (1 - lr) * matTheta_Rte[usr_idx,:] + lr * broadcast(+, repmat(sum(matBeta[itm_idx,:], 1), m, 1), matEpsilon[usr_idx])
  matTheta[usr_idx,:] = matTheta_Shp[usr_idx,:] ./ matTheta_Rte[usr_idx,:]

  matBeta_Rte[itm_idx, :] = (1 - lr) * matBeta_Rte[itm_idx, :] + lr * broadcast(+, scale * sum(matTheta[usr_idx,:], 1), matEta[itm_idx])
  matBeta[itm_idx, :] = matBeta_Shp[itm_idx, :] ./ matBeta_Rte[itm_idx, :]


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

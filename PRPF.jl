include("user_preference_train.jl")
include("SVI_PF.jl")

function sample_data(M::Int64, N::Int64, usr_batch_size::Int64)
  if usr_batch_size == M
    usr_idx = collect(1:M);
    itm_idx = collect(1:N);
    deleteat!(usr_idx, usr_zeros);
    deleteat!(itm_idx, itm_zeros);
    usr_idx_len = length(usr_idx);
    itm_idx_len = length(itm_idx);
  else
    usr_idx = randsample(M, usr_batch_size);
    deleteat!(usr_idx, sum(matX_train[usr_idx,:],2)==0);
    itm_idx = find(sum(matX_train[usr_idx, :], 1)>0);
    usr_idx_len = length(usr_idx);
    itm_idx_len = length(itm_idx);
  end
  return usr_idx, itm_idx, usr_idx_len, itm_idx_len
end


function PRPF(K::Float64, C::Float64, alpha::Float64=1000., delta::Float64=1., kappa::Float64=0.5, usr_batch_size::Int32
              topK = Array{Int16,1}, test_step::Int16=10, check_step::Int16=10, MaxItr::Int32=100,
              matX_train::SparseMatrixCSC{Float64,2}, matX_test::SparseMatrixCSC{Float64,2}, matX_valid::SparseMatrixCSC{Float64,2})
  M = size(matX_train, 1);
  N = size(matX_train, 2);
  usr_zeros = sum(matX, 2) .== 0;
  itm_zeros = sum(matX, 1) .== 0;

  IsConverge = false;
  itr = 0;

  #
  # Initialization
  #
  # Initialize matEpsilon
  matEpsilon_Shp = ini_scale * rand(M, 1) + b;
  matEpsilon_Rte = ini_scale * rand(M, 1) + c;
  matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte;

  # Initialize matEta
  matEta_Shp = ini_scale * rand(N, 1) + e;
  matEta_Rte = ini_scale * rand(N, 1) + f;
  matEta = matEta_Shp ./ matEta_Rte;

  # Initialize matBeta
  matBeta_Shp = ini_scale * rand(N, K) + d;
  matBeta_Rte = ini_scale * rand(N, K)
  for k=1:K
    this.matBeta_Rte[:,k] += matEta;
  end
  matBeta = matBeta_Shp ./ matBeta_Rte;
  matBeta_Shp[find(itm_zeros), :] = 0;
  matBeta_Rte[find(itm_zeros), :] = 0;
  matBeta[find(itm_zeros), :] = 0;

  # Initialize matTheta
  matTheta_Shp = ini_scale * rand(M, K) + a;
  matTheta_Rte = ini_scale * rand(M, K);
  for k=1:K
    this.matTheta_Rte[:,k] += matEpsilon;
  end
  matTheta = matTheta_Shp ./ matTheta_Rte;
  matTheta_Shp[find(usr_zeros),:] = 0;
  matTheta_Rte[find(usr_zeros),:] = 0;
  matTheta[find(usr_zeros),:] = 0;

  # Initialize matX_predict
  this.matX_predict = (matTheta[1,:] * matBeta[1,:]') .* (matX > 0);

  while IsConverge == false && itr < MaxItr
    itr += 1;

    #
    # Set the learning rate
    # ref: Content-based recommendations with Poisson factorization. NIPS, 2014
    #
    if usr_batch_size == M
        lr = 1.;
    else
        offset = 1.;
        lr = (offset + itr) ^ -kappa;
    end

    #
    # Sample data
    #
    usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, usr_zeros, itm_zeros);

    prior_X = (matTheta[usr_idx,:] * matBeta[itm_idx, :]') .* (matX_train[usr_idx,itm_idx]>0);
    predict_X = matX_predict[usr_idx, itm_idx] .* (matX_train[usr_idx,itm_idx]>0);

    for u = 1:usr_idx_len
      solution_xui_xuj2 = user_preference_train(vec_prior_X_u, vec_predict_X_u, vec_matX_u, delta, C, alpha)
    end

    SVI_PF(lr, M, N::UInt64, K, ini_scale, usr_idx, itm_idx, predict_X,
                                   matTheta, matTheta_Shp, matTheta_Rte, matBeta, matBeta_Shp, matBeta_Rte,
                                   matEpsilon, matEpsilon_Shp, matEpsilon_Rte, matEta, matEta_Shp, matEta_Rte;

  end

end

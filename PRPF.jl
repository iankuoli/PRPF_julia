include("user_preference_train.jl")
include("SVI_PF.jl")
include("evaluate.jl")
include("sample_data.jl")

A = sparse([0 0 0 5; 3 0 0 0; 0 2 4 0; 0 0 0 0])
usr_zeros = (sum(A, 2) .== 0)[:]

function PRPF(K::Float64, C::Float64, M::Int64, N::Int64,
              matX_train::SparseMatrixCSC{Float64,Int64}, matX_test::SparseMatrixCSC{Float64,Int64}, matX_valid::SparseMatrixCSC{Float64,Int64},
              alpha::Float64=1000., delta::Float64=1., kappa::Float64=0.5, usr_batch_size::Int32=0, MaxItr::Int32=100,
              topK::Array{Int16,1} = [10], test_step::Int16=10, check_step::Int16=10)

  usr_batch_size == 0? usr_batch_size = M:usr_batch_size;
  usr_zeros = (sum(matX, 2) .== 0)[:];
  itm_zeros = (sum(matX, 1) .== 0)[:];

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
    usr_batch_size == M? lr = 1.:(1. + itr) ^ -kappa;

    #if usr_batch_size == M
    #    lr = 1.;
    #else
    #    offset = 1.;
    #    lr = (offset + itr) ^ -kappa;
    #end

    #
    # Sample data
    #
    usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros);

    #
    # Estimate prediction \mathbf{x}_{ui}
    #
    prior_X = (matTheta[usr_idx,:] * matBeta[itm_idx, :]') .* (matX_train[usr_idx,itm_idx]>0);
    predict_X = matX_predict[usr_idx, itm_idx] .* (matX_train[usr_idx,itm_idx]>0);
    for u = 1:usr_idx_len
      vec_prior_X_u = prior_X[u,:];
      vec_predict_X_u = predict_X[u,:]
      vec_matX_u = matX_train[usr_idx[u], itm_idx];
      predict_X[u,:] = user_preference_train(vec_prior_X_u, vec_predict_X_u, vec_matX_u, delta, C, alpha)
    end

    #
    # Update latent variables
    #
    SVI_PF(lr, M, N, K, ini_scale, usr_idx, itm_idx, predict_X,
           matTheta, matTheta_Shp, matTheta_Rte, matBeta, matBeta_Shp, matBeta_Rte,
           matEpsilon, matEpsilon_Shp, matEpsilon_Rte, matEta, matEta_Shp, matEta_Rte);

    #
    # Validation
    #
    if mod(i, check_step) == 0 && check_step > 0
      valid_precision, valid_recall, Vlog_likelihood = evaluate(matX_valid, matX_train, matTheta, matBeta);
    end
  end

end

include("user_preference_train.jl")
include("SVI_PF.jl")
include("evaluate.jl")
include("sample_data.jl")

function PRPF(K::Int64, C::Float64, M::Int64, N::Int64, prior::Tuple{Float64,Float64,Float64,Float64,Float64,Float64}, ini_scale::Float64,
              matX_train::SparseMatrixCSC{Float64,Int64}, matX_test::SparseMatrixCSC{Float64,Int64}, matX_valid::SparseMatrixCSC{Float64,Int64},
              usr_batch_size::Int64=0, MaxItr::Int64=100, topK::Array{Int64,1} = [10], test_step::Int64=10, check_step::Int64=10,
              alpha::Float64=1000., delta::Float64=1., kappa::Float64=0.5)

  usr_batch_size == 0? usr_batch_size = M:usr_batch_size;

  usr_zeros = find((sum(matX_train, 2) .== 0)[:]);
  itm_zeros = find((sum(matX_train, 1) .== 0)[:]);

  (a,b,c,d,e,f) = prior;

  IsConverge = false;
  itr = 0;
  lr = 0.;

  valid_precision = zeros(Float64, length(topK));
  valid_recall = zeros(Float64, length(topK));
  Vlog_likelihood = 0.;

  #
  # Initialization
  #
  # Initialize matEpsilon
  matEpsilon_Shp = ini_scale * rand(M) + b;
  matEpsilon_Rte = ini_scale * rand(M) + c;
  matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte;

  # Initialize matEta
  matEta_Shp = ini_scale * rand(N) + e;
  matEta_Rte = ini_scale * rand(N) + f;
  matEta = matEta_Shp ./ matEta_Rte;

  # Initialize matBeta
  matBeta_Shp = ini_scale * rand(N, K) + d;
  matBeta_Rte = ini_scale * rand(N, K)
  for k=1:K
    matBeta_Rte[:,k] += matEta;
  end
  matBeta = matBeta_Shp ./ matBeta_Rte;
  matBeta_Shp[itm_zeros, :] = 0;
  matBeta_Rte[itm_zeros, :] = 0;
  matBeta[itm_zeros, :] = 0;

  # Initialize matTheta
  matTheta_Shp = ini_scale * rand(M, K) + a;
  matTheta_Rte = ini_scale * rand(M, K);
  for k=1:K
    matTheta_Rte[:,k] += matEpsilon;
  end
  matTheta = matTheta_Shp ./ matTheta_Rte;
  matTheta_Shp[usr_zeros,:] = 0;
  matTheta_Rte[usr_zeros,:] = 0;
  matTheta[usr_zeros,:] = 0;

  # Initialize matX_predict
  matX_predict = (matTheta[1,:]' * matBeta[1,:])[1] * (matX_train .> 0);

  while IsConverge == false && itr < MaxItr
    itr += 1;
    @printf("Step: %d \n", itr);

    #
    # Set the learning rate
    # ref: Content-based recommendations with Poisson factorization. NIPS, 2014
    #
    usr_batch_size == M? lr = 1.:(1. + itr) ^ -kappa;

    #
    # Sample data
    #
    usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros);

    #
    # Estimate prediction \mathbf{x}_{ui}
    #
    subPrior_X = (matTheta[usr_idx,:] * matBeta[itm_idx, :]') .* (matX_train[usr_idx,itm_idx] .> 0);
    subPredict_X = sparse(matX_predict[usr_idx, itm_idx] .* (matX_train[usr_idx,itm_idx] .> 0));
    for u = 1:usr_idx_len
      u_idx = usr_idx[u];
      (js, vs) = findnz(matX_train[u_idx, itm_idx]);

      vec_subPrior_X_u = full(subPrior_X[u, js]);
      vec_subPredict_X_u = full(subPredict_X[u, js]);
      vec_subMatX_u = full(matX_train[u_idx, itm_idx[js]]);

      # prediction w.r.t user $u$ by pair-wise LTR
      subPredict_X[u, js] = user_preference_train_pw(vec_subPrior_X_u, vec_subPredict_X_u, vec_subMatX_u, C, alpha, delta);
    end

    @printf("subPredict_X: ( %d , %d ) , nnz = %d \n", size(subPredict_X,1), size(subPredict_X,2), countnz(subPredict_X));

    #
    # Update latent variables
    #
    SVI_PF(lr, M, N, K, usr_batch_size, usr_idx, itm_idx, subPredict_X, matX_train,
           matTheta, matTheta_Shp, matTheta_Rte,
           matBeta, matBeta_Shp, matBeta_Rte,
           matEpsilon, matEpsilon_Shp, matEpsilon_Rte,
           matEta, matEta_Shp, matEta_Rte, prior);

    println(matTheta[1,:]);

    #
    # Validation
    #
    if mod(itr, check_step) == 0 && check_step > 0
      println("Validation ... ");
      valid_precision, valid_recall, Vlog_likelihood = evaluate(matX_valid, matX_train, matTheta, matBeta, topK, C, alpha);
      println("validation precision: " * string(valid_precision));
    end
  end

  return valid_precision, valid_recall, Vlog_likelihood,
         matTheta, matTheta_Shp, matTheta_Rte,
         matBeta, matBeta_Shp, matBeta_Rte,
         matEpsilon, matEpsilon_Shp, matEpsilon_Rte,
         matEta, matEta_Shp, matEta_Rte

end


# X =  sparse([5. 4 3 0 0 0 0 0;
#              3. 4 5 0 0 0 0 0;
#              0  0 0 3 3 4 0 0;
#              0  0 0 5 4 5 0 0;
#              0  0 0 0 0 0 5 4;
#              0  0 0 0 0 0 3 4;
#              0  0 0 0 0 0 0 0])
# matTheta = [1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 0]
# matBeta = [4 0 0; 4 0 0; 4 0 0; 0 4 0; 0 3 0; 0 5 0; 0 0 4; 0 0 4]
# usr_idx = [1,2,3,4]
# itm_idx = [1,2,3,4,5,6]
# TT = X[usr_idx,itm_idx] .> 0
# prior_X = sparse((matTheta[usr_idx,:] * matBeta[itm_idx, :]') .* (X[usr_idx,itm_idx] .> 0))

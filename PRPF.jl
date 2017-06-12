include("SVI_PF2.jl")
include("evaluate.jl")
include("sample_data.jl")
include("model_io.jl")


function PRPF(dataset::String, model_type::String, K::Int64, C::Float64, M::Int64, N::Int64,
              matX_train::SparseMatrixCSC{Float64,Int64}, matX_test::SparseMatrixCSC{Float64,Int64}, matX_valid::SparseMatrixCSC{Float64,Int64},
              prior::Tuple{Float64,Float64,Float64,Float64,Float64,Float64}=(0.3,0.3,0.3,0.3,0.3,0.3),
              ini_scale::Float64=0.003, usr_batch_size::Int64=0, MaxItr::Int64=100, topK::Array{Int64,1} = [10],
              test_step::Int64=10, check_step::Int64=10, alpha::Float64=200., delta::Float64=1., kappa::Float64=0.5)

  #usr_batch_size == 0? usr_batch_size = M:usr_batch_size

  usr_zeros = find((sum(matX_train, 2) .== 0)[:])
  itm_zeros = find((sum(matX_train, 1) .== 0)[:])

  (a,b,c,d,e,f) = prior

  IsConverge = false
  itr = 0
  lr = 0.

  valid_precision = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  valid_recall = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  Vlog_likelihood = zeros(Float64, Int(ceil(MaxItr/check_step)))

  test_precision = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  test_recall = zeros(Float64, Int(ceil(MaxItr/check_step)), length(topK))
  Tlog_likelihood = zeros(Float64, Int(ceil(MaxItr/check_step)))

  #
  # Initialization
  #
  # Initialize matEpsilon
  matEpsilon_Shp = ini_scale * rand(M) + b
  matEpsilon_Rte = ini_scale * rand(M) + c
  matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte

  # Initialize matEta
  matEta_Shp = ini_scale * rand(N) + e
  matEta_Rte = ini_scale * rand(N) + f
  matEta = matEta_Shp ./ matEta_Rte

  # Initialize matBeta
  matBeta_Shp = ini_scale * rand(N, K) + d
  matBeta_Rte = ini_scale * rand(N, K)
  for k=1:K
    matBeta_Rte[:,k] += matEta
  end
  matBeta = matBeta_Shp ./ matBeta_Rte
  matBeta_Shp[itm_zeros, :] = 0
  matBeta_Rte[itm_zeros, :] = 0
  matBeta[itm_zeros, :] = 0

  # Initialize matTheta
  matTheta_Shp = ini_scale * rand(M, K) + a
  matTheta_Rte = ini_scale * rand(M, K)
  for k=1:K
    matTheta_Rte[:,k] += matEpsilon
  end
  matTheta = matTheta_Shp ./ matTheta_Rte
  matTheta_Shp[usr_zeros,:] = 0
  matTheta_Rte[usr_zeros,:] = 0
  matTheta[usr_zeros,:] = 0

  # Initialize matX_predict
  matX_predict = sparse(findn(matX_train)..., (matTheta[1,:]' * matBeta[1,:])[1] * ones(nnz(matX_train)), M, N)

  if usr_batch_size == M || usr_batch_size == 0
    usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros)
    sampled_i, sampled_j = findn(matX_train[usr_idx, itm_idx])
  end

  while IsConverge == false && itr < MaxItr
    itr += 1;
    @printf("\nStep: %d \n", itr)

    #
    # Set the learning rate
    # ref: Content-based recommendations with Poisson factorization. NIPS, 2014
    #
    usr_batch_size == M? lr = 1.:(1. + itr) ^ -kappa

    #
    # Sample data
    #
    if usr_batch_size != M && usr_batch_size != 0
      usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros)
      sampled_i, sampled_j = findn(matX_train[usr_idx, itm_idx])
    end

    #
    # Estimate prediction \mathbf{x}_{ui}
    #
    tmp_v = zeros(nnz(matX_train))
    @time @fastmath for entry_itr = 1:length(sampled_i)
      i_idx = sampled_i[entry_itr]
      j_idx = sampled_j[entry_itr]
      tmp_v = infer_entry(matTheta, matBeta, i_idx, j_idx)
    end
    subPrior_X = sparse(sampled_i, sampled_j, tmp_v, length(usr_idx), length(itm_idx))

    if model_type == "HPF"
      subPredict_X = matX_train[usr_idx, itm_idx]
    else
      subPredict_X = matX_predict[usr_idx, itm_idx]

      for u = 1:usr_idx_len
        u_idx = usr_idx[u]
        (js, vs) = findnz(matX_train[u_idx, itm_idx])

        if length(js) < 2
          subPredict_X[u, :] = subPrior_X[u, :]
          continue
        end

        vec_subPrior_X_u = full(subPrior_X[u, js])[:]
        vec_subPredict_X_u = full(subPredict_X[u, js])[:]
        vec_subMatX_u = full(matX_train[u_idx, itm_idx[js]])[:]

        if model_type == "PairPRPF"
          # Prediction w.r.t user $u$ by pair-wise LTR
          subPredict_X[u, js] = @fastmath user_preference_train_pw(vec_subPrior_X_u, vec_subPredict_X_u, vec_subMatX_u, C, alpha, delta)

        elseif model_type == "LuceExpPRPF"
          # Prediction w.r.t user $u$ by luce-based list-wise LTR with exponential transformation
          subPredict_X[u, js] = @fastmath user_preference_train_luce(vec_subPrior_X_u, vec_subPredict_X_u, vec_subMatX_u, C, alpha, delta, "exp")

        elseif model_type == "LuceLinearPRPF"
          # Prediction w.r.t user $u$ by luce-based list-wise LTR with linear transformation
          subPredict_X[u, js] = @fastmath user_preference_train_luce(vec_subPrior_X_u, vec_subPredict_X_u, vec_subMatX_u, C, alpha, delta, "linear")

        end
      end

      matX_predict[usr_idx,itm_idx] = (1-lr) * matX_predict[usr_idx, itm_idx] + lr * subPredict_X
      subPredict_X = matX_predict[usr_idx, itm_idx]

      #println(matX_predict[32,:])
      #println(matX_train[32,:])
    end

    @printf("subPredict_X: ( %d , %d ) , nnz = %d , lr = %f \n", size(subPredict_X,1), size(subPredict_X,2), countnz(subPredict_X), lr)

    #
    # Update latent variables
    #
    SVI_PF2(lr, M, N, K, usr_batch_size, usr_idx, itm_idx, subPredict_X, matX_train,
            matTheta, matTheta_Shp, matTheta_Rte,
            matBeta, matBeta_Shp, matBeta_Rte,
            matEpsilon, matEpsilon_Shp, matEpsilon_Rte,
            matEta, matEta_Shp, matEta_Rte, prior);

    #
    # Validation
    #
    if mod(itr, check_step) == 0 && check_step > 0
      println("Validation ... ")
      indx = Int(itr / check_step)
      valid_precision[indx,:], valid_recall[indx,:], Vlog_likelihood[indx,:] = evaluate(matX_valid, matX_train, matTheta, matBeta, topK, C, alpha)
      println("validation precision: " * string(valid_precision[indx,:]))
      println("validation recall: " * string(valid_recall[indx,:]))

      #
      # Check whether the step performs the best. If yes, run testing and save model
      #
      if findmax(valid_precision[:,1])[2] == Int(itr / check_step)
        # Testing
        println("Testing ... ")
        indx = Int(itr / check_step)
        test_precision[indx,:], test_recall[indx,:], Tlog_likelihood[indx,:] = evaluate(matX_test, matX_train, matTheta, matBeta, topK, C, alpha)
        println("testing precision: " * string(test_precision[indx,:]))
        println("testing recall: " * string(test_recall[indx,:]))

        # Save model
        file_name = string(model_type, "_K", K, "_", string(now())[1:10])
        write_model(file_name, matTheta, matBeta, matEpsilon, matEta, prior, C, delta, alpha)
      end
    end

    #
    # Testing
    #
    # if mod(itr, test_step) == 0 && test_step > 0
    #   println("Testing ... ")
    #   indx = Int(itr / test_step)
    #   test_precision[indx,:], test_recall[indx,:], Tlog_likelihood[indx,:] = evaluate(matX_test, matX_train, matTheta, matBeta, topK, C, alpha)
    #   println("testing precision: " * string(test_precision[indx,:]))
    #   println("testing recall: " * string(test_recall[indx,:]))
    # end
  end

  #print(matTheta);

  return test_precision, test_recall, Vlog_likelihood,
         valid_precision, valid_recall, Vlog_likelihood,
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



#
# Test Update
#
# A = sparse([3. 0 0 8; 4 5 0 0; 0 0 6 8; 0 8 6 0])
# matTheta_Shp = [10. 1; 1 8; 8 1; 1 10] / 10
# matTheta_Rte = [2. 2; 2 2; 3 3; 2 2] / 10
# matTheta = matTheta_Shp ./ matTheta_Rte
#
# matBeta_Shp = [10. 1; 1 8; 8 1; 1 10] / 10
# matBeta_Rte = [2. 2; 2 2; 3 3; 2 2] / 10
# matBeta = matBeta_Shp ./ matBeta_Rte
#
# matEpsilon_Shp = [0.1, 0.3, 0.4, 0.5] / 10 + 0.3
# matEpsilon_Rte = [0.3, 0.2, 0.6, 0.2] / 10 + 0.3
# matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte
#
# # Initialize matEta
# matEta_Shp = [0.6, 0.4, 0.3, 0.1] / 10 + 0.3 + 0.3;
# matEta_Rte = [0.2, 0.5, 0.3, 0.4] + 0.3;
# matEta = matEta_Shp ./ matEta_Rte
#
# prior = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3)
#
# usr_batch_size = 4;
# usr_idx = [1,2,3,4]
# itm_idx = [1,2,3,4]
#
# for i = 1:50
#   SVI_PF(1., 4, 4, 2, usr_batch_size, usr_idx, itm_idx, A, A,
#          matTheta, matTheta_Shp, matTheta_Rte,
#          matBeta, matBeta_Shp, matBeta_Rte,
#          matEpsilon, matEpsilon_Shp, matEpsilon_Rte,
#          matEta, matEta_Shp, matEta_Rte, prior);
#
#
# end
#
# matTheta
#
#
#
#
#
# matBeta

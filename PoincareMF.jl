include("evaluate.jl")
include("sample_data.jl")
include("model_io.jl")

function distance_Poincare(rep1::Array{Float64,1}, rep2::Array{Float64,1})
  return acosh( 1 + 2 * ( norm(rep1 - rep2) / (norm(rep1) * norm(rep2)) )^2 )
end


function update_matTheta(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                         vecGamma::Array{Float64,1}, vecDelta::Array{Float64,1},
                         sampled_i::Array{Int64,1}, sampled_j::Array{Int64,1}, sampled_v::Array{Float64,1},
                         K::Int64, usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})
  vec_theta_norm = zeros(usr_idx_len)
  vec_beta_norm = zeros(itm_idx_len)
  for u_itr=1:usr_idx_len
    vec_theta_norm[u_itr] = norm(matTheta[usr_idx[u_itr],:])
  end
  for i_itr=1:itm_idx_len
    vec_beta_norm[i_itr] = norm(matBeta[itm_idx[i_itr],:])
  end

  mat_partial_L_by_theta = zeros(Float64, usr_idx_len, K)
  vec_partial_L_by_gamma = zeros(Float64, usr_idx_len)
  for itr=1:length(sampled_i)
    u = sampled_i[itr]
    i = sampled_j[itr]
    u_id = usr_idx[u]
    i_id = itm_idx[i]

    norm_diff = norm(matTheta[u_id,:] - matBeta[i_id,:])
    dot_theta_beta = dot(matTheta[u_id,:], matBeta[i_id,:])

    # Compute vec_partial_L_by_d
    f_ui = exp( acosh( 1 + 2 * ( norm_diff / (vec_theta_norm[u] * vec_beta_norm[i]) )^2 ) + vecGamma[u] + vecDelta[i] )
    #f_ui = exp(distance_Poincare(matTheta[usr_idx[u],:], matBeta[itm_idx[i],:]) + vecGamma[u] + vecDelta[i])
    vec_partial_L_by_d = (1 / (-1 + f_ui) + (1 + alpha * sampled_v[itr]) / (1 + f_ui)) * f_ui

    # Compute vec_partial_d_by_theta
    a = 1 - vec_theta_norm[u] ^ 2
    b = 1 - vec_beta_norm[i] ^ 2
    c = 1 + 2/(a*b) * norm_diff^2
    vec_partial_d_by_theta_u = 4/(b * sqrt(c^2 - 1)) * ( (vec_beta_norm[i]^2 - 2 * dot_theta_beta + 1) / a^2 * matTheta[u_id,:] - matBeta[i_id,:] / a )

    mat_partial_L_by_theta[u,:] += vec_partial_L_by_d * vec_partial_d_by_theta_u
    vec_partial_L_by_gamma[u] += vec_partial_L_by_d
  end

  for u = 1:usr_idx_len
    u_id = usr_idx[u]
    matTheta[u_id,:] -= lr * (1 - vec_theta_norm[u]^2)^2 / 4 * mat_partial_L_by_theta[u,:]
    norm_tmp = norm(matTheta[u_id,:])
    if norm_tmp >= 1
      matTheta[u_id,:] = matTheta[u_id,:] / norm_tmp - 10e-6
    end
  end

  vecGamma[usr_idx] -= lr * vec_partial_L_by_gamma

  return nothing
end


function update_matBeta(matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                        vecGamma::Array{Float64,1}, vecDelta::Array{Float64,1},
                        sampled_i::Array{Int64,1}, sampled_j::Array{Int64,1}, sampled_v::Array{Float64,1},
                        K::Int64, usr_idx_len::Int64, itm_idx_len::Int64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1})
  vec_theta_norm = zeros(usr_idx_len)
  vec_beta_norm = zeros(itm_idx_len)
  for u_itr=1:usr_idx_len
    vec_theta_norm[u_itr] = norm(matTheta[usr_idx[u_itr],:])
  end
  for i_itr=1:itm_idx_len
    vec_beta_norm[i_itr] = norm(matBeta[itm_idx[i_itr],:])
  end

  mat_partial_L_by_beta = zeros(Float64, itm_idx_len, K)
  vec_partial_L_by_delta = zeros(Float64, itm_idx_len)
  for itr=1:length(sampled_i)
    u = sampled_i[itr]
    i = sampled_j[itr]
    u_id = usr_idx[u]
    i_id = itm_idx[i]

    norm_diff = norm(matTheta[u_id,:] - matBeta[i_id,:])
    dot_theta_beta = dot(matTheta[u_id,:], matBeta[i_id,:])

    # Compute vec_partial_L_by_d
    f_ui = exp( acosh( 1 + 2 * ( norm_diff / (vec_theta_norm[u] * vec_beta_norm[i]) )^2 ) + vecGamma[u] + vecDelta[i] )
    #f_ui = exp(distance_Poincare(matTheta[usr_idx[u],:], matBeta[itm_idx[i],:]) + vecGamma[u] + vecDelta[i])
    vec_partial_L_by_d = (1 / (-1 + f_ui) + (1 + alpha * sampled_v[itr]) / (1 + f_ui)) * f_ui

    # Compute vec_partial_d_by_theta
    a = 1 - vec_beta_norm[i] ^ 2
    b = 1 - vec_theta_norm[u] ^ 2
    c = 1 + 2/(a*b) * norm_diff^2
    vec_partial_d_by_beta_u = 4/(b * sqrt(c^2 - 1)) * ( (vec_theta_norm[u]^2 - 2 * dot_theta_beta + 1) / a^2 * matBeta[i_id,:] - matTheta[u_id,:] / a )

    mat_partial_L_by_beta[i,:] += vec_partial_L_by_d * vec_partial_d_by_beta_u
    vec_partial_L_by_delta[i] += vec_partial_L_by_d
  end

  for i = 1:itm_idx_len
    i_id = itm_idx[i]
    matBeta[i_id,:] -= lr * (1 - vec_beta_norm[i]^2)^2 / 4 * mat_partial_L_by_beta[i,:]
    norm_tmp = norm(matBeta[i_id,:])
    if norm_tmp >= 1
      matBeta[i_id,:] = matBeta[i_id,:] / norm_tmp - 10e-6
    end
  end

  vecDelta[itm_idx] -= lr * vec_partial_L_by_delta

  return nothing
end


function PoincareMF(model_type::String, K::Int64, M::Int64, N::Int64,
                    matX_train::SparseMatrixCSC{Float64,Int64}, matX_test::SparseMatrixCSC{Float64,Int64}, matX_valid::SparseMatrixCSC{Float64,Int64},
                    ini_scale::Float64=0.003, alpha::Float64=1.0, lr::Float64=0.01, usr_batch_size::Int64=0, MaxItr::Int64=100, topK::Array{Int64,1} = [10],
                    test_step::Int64=0, check_step::Int64=5)

  usr_batch_size == 0? usr_batch_size = M:usr_batch_size

  usr_zeros = find((sum(matX_train, 2) .== 0)[:])
  itm_zeros = find((sum(matX_train, 1) .== 0)[:])

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
  # Initialize matTheta
  matTheta = ini_scale * rand(M, K)
  matTheta[usr_zeros,:] = 0

  # Initialize matBeta
  matBeta = ini_scale * rand(N, K)
  matBeta[itm_zeros, :] = 0

  #Initialize Biases
  vecGamma = ini_scale * rand(M)
  vecGamma[usr_zeros] = 0
  vecDelta = ini_scale * rand(N)
  vecDelta[itm_zeros] = 0

  if usr_batch_size == M || usr_batch_size == 0
    usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros)
    sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])
  end

  while IsConverge == false && itr < MaxItr
    itr += 1;
    @printf("\nStep: %d \n", itr)

    #
    # Set the learning rate
    # ref: "Poincar\'e Embeddings for Learning Hierarchical Representations.". arXiv by FAIR, 2017
    #
    usr_batch_size == M? lr = 1.:(1. + itr) ^ -kappa

    #
    # Sample data
    #
    if usr_batch_size != M && usr_batch_size != 0
      usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros)
      sampled_i, sampled_j, sampled_v = findnz(matX_train[usr_idx, itm_idx])
    end


    #
    # Update parameters by alternating update -----------------------------------
    #

    #
    # Update matTheta & vecGamma
    #
    update_matTheta(matTheta, matBeta, vecGamma, vecDelta, sampled_i, sampled_j, sampled_v, K, usr_idx_len, itm_idx_len, usr_idx, itm_idx)
    update_matBeta(matTheta, matBeta, vecGamma, vecDelta, sampled_i, sampled_j, sampled_v, K, usr_idx_len, itm_idx_len, usr_idx, itm_idx)

    #
    # Validation
    #
    if mod(itr, check_step) == 0 && check_step > 0
      println("Validation ... ")
      indx = Int(itr / check_step)
      valid_precision[indx,:], valid_recall[indx,:], Vlog_likelihood[indx,:] = evaluate(matX_valid, matX_train, matTheta, matBeta, topK, C, alpha)
      println("QQ")
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
        write_model(file_name, matTheta, matBeta, vecGamma, vecDelta, lr)
      end
    end


    #
    # Testing
    #
    if test_step > 0 && mod(itr, test_step) == 0
      println("Testing ... ")
      indx = Int(itr / test_step)
      test_precision[indx,:], test_recall[indx,:], Tlog_likelihood[indx,:] = evaluate(matX_test, matX_train, matTheta, matBeta, topK, C, alpha)
      println("testing precision: " * string(test_precision[indx,:]))
      println("testing recall: " * string(test_recall[indx,:]))
    end
  end

  return test_precision, test_recall, Vlog_likelihood,
         valid_precision, valid_recall, Vlog_likelihood,
         matTheta, vecGamma,
         matBeta, vecDelta
end

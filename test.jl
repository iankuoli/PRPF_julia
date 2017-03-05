include("LoadData.jl")
include("PRPF.jl")

function train_filepath(data_name::String, env::Int64)
  if env == 1
    #
    # Macbook in my house
    #
    if data_name == "Last.fm1K"
      # Last.fm1K
      training_path = "/Users/iankuoli/Dataset/LastFm1K_train.csv";
      testing_path = "/Users/iankuoli/Dataset/LastFm1K_test.csv";
      validation_path = "/Users/iankuoli/Dataset/LastFm1K_valid.csv";

    elseif data_name == "Last.fm2K"
      # Last.fm2K
      training_path = "/Users/iankuoli/Dataset/LastFm_train.csv";
      testing_path = "/Users/iankuoli/Dataset/LastFm_test.csv";
      validation_path = "/Users/iankuoli/Dataset/LastFm_valid.csv";

    elseif data_name == "MovieLens100K"
      # MovieLens100K
      training_path = "/Users/iankuoli/Dataset/MovieLens100K_train.csv";
      testing_path = "/Users/iankuoli/Dataset/MovieLens100K_test.csv";
      validation_path = "/Users/iankuoli/Dataset/MovieLens100K_valid.csv";

    elseif data_name == "MovieLens1M"
      # MovieLens1M
      training_path = "/Users/iankuoli/Dataset/MovieLens1M_train.csv";
      testing_path = "/Users/iankuoli/Dataset/MovieLens1M_test.csv";
      validation_path = "/Users/iankuoli/Dataset/MovieLens1M_valid.csv";
    end
  elseif env == 2
    #
    # CentOS in my office
    #
    if data_name == "Last.fm1K"
      # Last.fm1K
      training_path = "/Users/iankuoli/Dataset/LastFm1K_train.csv";
      testing_path = "/Users/iankuoli/Dataset/LastFm1K_test.csv";
      validation_path = "/Users/iankuoli/Dataset/LastFm1K_valid.csv";

    elseif data_name == "Last.fm2K"
      # Last.fm2K
      training_path = "/home/ian/Dataset/LastFm_train.csv";
      testing_path = "/home/ian/Dataset/LastFm_test.csv";
      validation_path = "/home/ian/Dataset/LastFm_valid.csv";

    elseif data_name == "MovieLens100K"
      # MovieLens100K
      training_path = "/home/ian/Dataset/MovieLens100K_train.csv";
      testing_path = "/home/ian/Dataset/MovieLens100K_test.csv";
      validation_path = "/home/ian/Dataset/MovieLens100K_valid.csv";

    elseif data_name == "MovieLens1M"
      # MovieLens1M
      training_path = "/Users/iankuoli/Dataset/MovieLens1M_train.csv";
      testing_path = "/Users/iankuoli/Dataset/MovieLens1M_test.csv";
      validation_path = "/Users/iankuoli/Dataset/MovieLens1M_valid.csv";
    end
  end
  return training_path, testing_path, validation_path
end

function train_setting(data_name::String, model::String)
  # Initialize the paramters
  prior = 0.;
  ini_scale = 0.;
  batch_size = 0.;
  MaxItr = 0.;
  test_step = 0.;
  check_step = 0.;
  lr = 0.;
  lambda = 0.;
  lambda_Theta = 0.;
  lambda_Beta = 0.;
  lambda_B = 0.;

  if data_name == "Last.fm1K"
    # Last.fm1K
    if model == "PRPF" || model == "PF" || model == "pointPRPF"
      prior = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      MaxItr = 60;
      test_step = 20;
      check_step = 10;
    elseif model == "LogMF"
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      lr = 0.000001;
      lambda = 0;
      alpha = 1;
      test_step = 800;
      MaxItr = 5000;
      check_step = 400;
    elseif model == "ListPMF"
      lambda   = 0.001;
      lambda_Theta = 1;
      lambda_Beta = 1;
      lambda_B = 1;
      test_step = 5;
      MaxItr = 3000;
      check_step = 10;
    elseif model == "BPR"
      lr = 0.2;       # learning rate
      lambda = 0;     # regularization weight
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      Itr_step = 2000;
      MaxItr = 2500 * Itr_step;
    end

  elseif data_name == "Last.fm2K"
    # Last.fm2K
    if model == "PRPF" || model == "PF" || model == "pointPRPF"
      prior = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      MaxItr = 60;
      test_step = 20;
      check_step = 10;
    elseif model == "LogMF"
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      lr = 0.000001;
      lambda = 0;
      alpha = 1;
      test_step = 800;
      MaxItr = 5000;
      check_step = 400;
    elseif model == "ListPMF"
      lambda   = 0.001;
      lambda_Theta = 1;
      lambda_Beta = 1;
      lambda_B = 1;
      test_step = 5;
      MaxItr = 3000;
      check_step = 10;
    elseif model == "BPR"
      lr = 0.2;       # learning rate
      lambda = 0;     # regularization weight
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      Itr_step = 2000;
      MaxItr = 2500 * Itr_step;
    end

  elseif data_name == "MovieLens100K"
    # MovieLens100K
    if model == "PRPF" || model == "PF" || model == "pointPRPF"
      prior = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size=943;
      MaxItr = 150;
      test_step = 5;
      check_step = 5;
    elseif model == "LogMF"
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      lr = 0.001;
      lambda = 0;
      alpha = 1;
      test_step = 50;
      MaxItr = 3000;
      check_step = 50;
    elseif model == "ListPMF"
      lambda   = 0.001;
      lambda_Theta = 0.01;
      lambda_Beta = 0.01;
      lambda_B = 0.01;
      test_step = 5;
      MaxItr = 200;
      check_step = 5;
    elseif model == "BPR"
      lr = 0.2;          # learning rate
      lambda = 0;        # regularization weight
      prior = [0.3 0.3];
      ini_scale = prior[1]/100;
      Itr_step = 2000;
      MaxItr = 2500 * Itr_step;
    end

  elseif data_name == "MovieLens1M"
    # MovieLens1M
    if model == "PRPF" || model == "PF" || model == "pointPRPF"
      prior = (0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 6040;
      MaxItr = 100;
      test_step = 5;
      check_step = 5;
    elseif model == "LogMF"
      prior = (0.3, 0.3);
      ini_scale = prior[1]/100;
      batch_size = 100;
      lr = 0.001;
      lambda = 0;
      alpha = 1;
      test_step = 1000;
      MaxItr = 50000;
      check_step = 500;
    elseif model == "ListPMF"
      lambda   = 0.01;
      lambda_Theta = 0.1;
      lambda_Beta = 0.1;
      lambda_B = 0.1;
      test_step = 5;
      MaxItr = 200;
      check_step = 5;
    elseif model == "BPR"
      lr = 0.1;         # learning rate
      lambda = 0.001;   # regularization weight
      prior = [0.3 0.3];
      ini_scale = prior[1]/100;
      Itr_step = 2000;
      MaxItr = 1000 * Itr_step;
    end

  end

  return prior, ini_scale, batch_size, MaxItr, test_step, check_step, lr,
        lambda, lambda_Theta, lambda_Beta, lambda_B
end

training_path, testing_path, validation_path = train_filepath("Last.fm2K", 1)

prior, ini_scale, batch_size, MaxItr, test_step, check_step, lr,
lambda, lambda_Theta, lambda_Beta, lambda_B = train_setting("Last.fm2K", "PRPF")

matX_train, matX_test, matX_valid, M, N = LoadUtilities(training_path, testing_path, validation_path)

prior
typeof(prior)




K = 100
topK = [5, 10, 15, 20]
C = mean(sum(matX_train .> 0, 2))
usr_batch_size = 0
ini_scale
PRPF(K, C, M, N, prior, ini_scale, matX_train, matX_test, matX_valid, usr_batch_size, MaxItr, topK, test_step, check_step)





alpha = 1000.
delta = 1.
kappa = 0.5
usr_batch_size == 0? usr_batch_size = M:usr_batch_size
usr_zeros = find((sum(matX_train, 2) .== 0)[:])
itm_zeros = find((sum(matX_train, 1) .== 0)[:])

(a,b,c,d,e,f) = prior

IsConverge = false
itr = 0
lr = 0.

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
matBeta_Shp[find(itm_zeros), :] = 0
matBeta_Rte[find(itm_zeros), :] = 0
matBeta[find(itm_zeros), :] = 0

# Initialize matTheta
matTheta_Shp = ini_scale * rand(M, K) + a
matTheta_Rte = ini_scale * rand(M, K)
for k=1:K
  matTheta_Rte[:,k] += matEpsilon;
end
matTheta = matTheta_Shp ./ matTheta_Rte
matTheta_Shp[find(usr_zeros),:] = 0
matTheta_Rte[find(usr_zeros),:] = 0
matTheta[find(usr_zeros),:] = 0

matX_predict = (matTheta[1,:]' * matBeta[1,:])[1] * (matX_train .> 0)
itr += 1
usr_batch_size == M? lr = 1.:(1. + itr) ^ -kappa

usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(M, N, usr_batch_size, matX_train, usr_zeros, itm_zeros)

subPrior_X = (matTheta[usr_idx,:] * matBeta[itm_idx, :]') .* (matX_train[usr_idx,itm_idx] .> 0)
subPredict_X = sparse(matX_predict[usr_idx, itm_idx] .* (matX_train[usr_idx,itm_idx] .> 0))
for u = 1:usr_idx_len
  u_idx = usr_idx[u];
  (js, vs) = findnz(matX_train[u_idx, itm_idx]);

  vec_subPrior_X_u = full(subPrior_X[u, js]);
  vec_subPredict_X_u = full(subPredict_X[u, js]);
  vec_subMatX_u = full(matX_train[u_idx, itm_idx[js]]);

  subPredict_X[u, js] = user_preference_train(vec_subPrior_X_u, vec_subPredict_X_u, vec_subMatX_u, delta, C, alpha);
end

tensorPhi = Update_tensorPhi(subPredict_X, matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:])
matTheta, matTheta_Shp, matTheta_Rte = Update_matTheta(M, N, K, usr_batch_size, lr, usr_idx, itm_idx,
                                                       matX_train, tensorPhi, matBeta, a, matEpsilon,
                                                       matTheta, matTheta_Shp, matTheta_Rte)
matBeta, matBeta_Shp, matBeta_Rte = Update_matBeta(M, N, K, usr_batch_size, lr, usr_idx, itm_idx,
                                                  matX_train, tensorPhi, matTheta, d, matEta,
                                                  matBeta, matBeta_Shp, matBeta_Rte)

matEpsilon, matEpsilon_Shp, matEpsilon_Rte = Update_matEpsilon(lr, usr_idx, itm_idx, matTheta, a, b, c,
                                                              matEpsilon, matEpsilon_Shp, matEpsilon_Rte)

matEta, matEta_Shp, matEta_Rte = Update_matEta(lr, usr_idx, itm_idx, matBeta, d, e, f,
                                              matEta, matEta_Shp, matEta_Rte)

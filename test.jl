include("LoadData.jl")
include("PRPF.jl")
include("conf.jl")


#
# Setting.
#
#dataset = "MovieLens1M"
#dataset = "MovieLens100K"
#dataset = "Lastfm1K"
#dataset = "Lastfm2K"
dataset = "Lastfm360K"
#dataset = "SmallToy"
env = 2
model_type = "PairPRPF"

#Ks = [5, 20, 50, 100, 150, 200]
Ks = [100]
topK = [5, 10, 15, 20]
usr_batch_size = 0
test_step = 2
check_step = 2
MaxItr = 40

if model_type == "PairPRPF"
  alpha = 1000.
elseif model_type == "LuceExpPRPF"
  alpha = 10. #40. #60.
elseif model_type == "LuceLinearPRPF"
  alpha = 10.
else
  alpha = 1.
end
#
# if env == 1
#   results_path = string("/Users/iankuoli/GitHub/PRPF_julia/results/", dataset, "_", model_type, "_alpha", Int(alpha), ".csv")
# elseif env == 2
#   results_path = joinpath(homedir(), "workspace", "julia", "PRPF_julia", "results", string(dataset, "_", model_type, "_alpha", Int(alpha), ".csv"))
#   #results_path = string("~/workspace/julia/PRPF_julia/results/", dataset, "_", model_type, "_alpha", Int(alpha), ".csv")
# end
results_path = joinpath("results", string(dataset, "_", model_type, "_alpha", Int(alpha), ".csv"))

#
# Initialize the hyper-parameters.
#
(prior, ini_scale, batch_size, MaxItr, test_step, check_step, lr, lambda, lambda_Theta, lambda_Beta, lambda_B) = train_setting(dataset, "PRPF")


#
# Load files to construct training set (utility matrices), validation set and test set.
#
training_path, testing_path, validation_path = train_filepath(dataset)
matX_train, matX_test, matX_valid, M, N = LoadUtilities(training_path, testing_path, validation_path)


#
# Training
#
listBestPrecisionNRecall = zeros(length(Ks), length(topK)*2)
C = nnz(matX_train) / size(matX_train,1)
for k = 1:length(Ks)
  K = Ks[k]
  test_precision, test_recall, Tlog_likelihood,
  valid_precision, valid_recall, Vlog_likelihood,
  matTheta, matTheta_Shp, matTheta_Rte,
  matBeta, matBeta_Shp, matBeta_Rte,
  matEpsilon, matEpsilon_Shp, matEpsilon_Rte,
  matEta, matEta_Shp, matEta_Rte = PRPF(dataset, model_type, K, C, M, N,
                                        matX_train, matX_test, matX_valid,
                                        prior, ini_scale, usr_batch_size, MaxItr, topK,
                                        test_step, check_step)

  (bestVal, bestIdx) = findmax(test_precision[:,1])
  listBestPrecisionNRecall[k,:] = [test_precision[bestIdx, :]; test_recall[bestIdx, :]]

  open(results_path, "a") do f
    writedlm(f, listBestPrecisionNRecall[k,:]')
  end
end
writedlm(results_path, listBestPrecisionNRecall)

listBestPrecisionNRecall





















#

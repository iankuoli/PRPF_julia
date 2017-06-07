include("LoadData.jl")
include("model_io.jl")
include("conf.jl")
include("evaluate.jl")

#
# Setting.
#
file_name = "PairPRPF_K100_2017-06-07"

#dataset = "MovieLens1M"
dataset = "MovieLens100K"
#dataset = "Lastfm1K"
#dataset = "Lastfm2K"
#dataset = "SmallToy"
env = 2
model_type = "PairPRPF"
topK = [5, 10, 15, 20]


#
# Load files to construct matrices.
#
training_path, testing_path, validation_path = train_filepath(dataset, env)
matX_train, matX_test, matX_valid, M, N = LoadUtilities(training_path, testing_path, validation_path)


#
# Load files to reconstruct the model.
#
matTheta, matBeta, matEpsilon, matEta, prior, C, delta, alpha = read_model(file_name)


#
# Evaluate the performace of the model.
#
test_precision, test_recall, Tlog_likelihood = evaluate(matX_test, matX_train, matTheta, matBeta, topK, C, alpha)





#

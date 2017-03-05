include("measure.jl")
include("inference.jl")

function evaluate(matX::SparseMatrixCSC{Float64, Int64}, matX_train::SparseMatrixCSC{Float64, Int64},
                  matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, topK::Array{Int64,1}, C::Float64, alpha::Float64)

  (vec_usr_idx, j, v) = findnz(sum(matX, 2));
  list_vecPrecision = zeros(length(topK));
  list_vecRecall = zeros(length(topK));
  log_likelihood = 0;
  step_size = 3000;
  denominator = 0;

  for j = 1:4#ceil(length(test_usr_idx)/step_size)
    range_step = collect((1 + (j-1) * step_size):min(j*step_size, length(vec_usr_idx)));

    # Compute the Precision and Recall
    matPredict = inference(range_step, matTheta, matBeta);
    matPredict -= matPredict .* (matX_train[vec_usr_idx[range_step], :] .> 0);
    (vec_precision, vec_recall) = compute_precNrec(matX[vec_usr_idx[range_step], :], matPredict, topK);

    list_vecPrecision += sum(vec_precision, 1)[:];
    list_vecRecall += sum(vec_recall, 1)[:];
    log_likelihood +=  LogPRPFObjFunc(C, alpha, matX[vec_usr_idx[range_step], :], matPredict) +
                        DistributionPoissonLogNZ(matX[vec_usr_idx[range_step], :], matPredict);
    denominator = denominator + length(range_step);
  end
  precision = list_vecPrecision / denominator;
  recall = list_vecRecall / denominator;
  log_likelihood = log_likelihood / countnz(matX);

  return precision, recall, log_likelihood
end

#
#  /// --- Unit test for function: evaluate() --- ///
#
# X =  sparse([5. 4 3 0 0 0 0 0;
#              3. 4 5 0 0 0 0 0;
#              0  0 0 3 3 4 0 0;
#              0  0 0 5 4 5 0 0;
#              0  0 0 0 0 0 5 4;
#              0  0 0 0 0 0 3 4])
# A = [1. 0 0; 1 0 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1]
# B = [4. 0 0; 4 0 0; 4 0 0; 0 4 0; 0 3 0; 0 5 0; 0 0 4; 0 0 4]
# XX = spzeros(6,8)
# topK = [1, 2]
# C = 1.
# alpha = 1000.
# precision, recall, log_likelihood = evaluate(X, XX, A, B, topK, C, alpha)

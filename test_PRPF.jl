delta = 1.

usr_batch_size == 0? usr_batch_size = M:usr_batch_size

K = Ks[1]

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

itr += 1
@printf("\nStep: %d \n", itr)

usr_batch_size == M? lr = 1.:(1. + itr) ^ -0.5

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

print(subPrior_X[1,1:5])
print(matX_predict[1,1:5])

subPredict_X = @parallel vcat for u = 1:usr_idx_len
 pred_Preference(model_type, usr_idx, itm_idx, u, C, alpha, delta, subPrior_X, matX_predict[usr_idx, itm_idx], matX_train)
end

matX_predict[usr_idx,itm_idx] = (1-lr) * matX_predict[usr_idx, itm_idx] + lr * subPredict_X
subPredict_X = matX_predict[usr_idx, itm_idx]

println(subPredict_X[1,1:5])
println(subPredict_X[1,1:5])







A = rand(10)

@everywhere function fff3(j::Int64)
  return vcat([j, j^2, j^3], [j, j+1, j+2], j^1, j^0.5)
end


fff3(4)

aaa = @parallel (+) for j = 1:10
  fff3(j)
end

aaa


aaa[end-1]



#

vcat([1,2,3], [4,5,6], 7, 8)

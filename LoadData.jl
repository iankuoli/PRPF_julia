function LoadUtilities(train_path::AbstractString, test_path::AbstractString, valid_path::AbstractString)
  #
  # Read file
  #

  if !isempty(train_path)
    matIndex_train = readdlm(train_path, ',');
  end

  if !isempty(test_path)
    matIndex_test = readdlm(test_path, ',');
  end

  if !isempty(valid_path)
    matIndex_valid = readdlm(valid_path, ',');
  end

  #
  # Compute: # of user, # of items
  #
  M = max(maximum(matIndex_train[:,1]), maximum(matIndex_test[:,1]), maximum(matIndex_valid[:,1]));
  N = max(maximum(matIndex_train[:,2]), maximum(matIndex_test[:,2]), maximum(matIndex_valid[:,2]));

  matX_train = sparse(convert(Array{Int64,1}, matIndex_train[:,1]), convert(Array{Int64,1}, matIndex_train[:,2]), matIndex_train[:,3], M, N);
  matX_test = sparse(convert(Array{Int64,1}, matIndex_test[:,1]), convert(Array{Int64,1}, matIndex_test[:,2]), matIndex_test[:,3], M, N);
  matX_valid = sparse(convert(Array{Int64,1}, matIndex_valid[:,1]), convert(Array{Int64,1}, matIndex_valid[:,2]), matIndex_valid[:,3], M, N);

  return matX_train, matX_test, matX_valid, convert(Int64, M), convert(Int64, N)
end

#
# // ----- Unit Test ----- //
#
# train_path = "/home/ian/Dataset/LastFm_train.csv";
# test_path = "/home/ian/Dataset/LastFm_test.csv";
# valid_path = "/home/ian/Dataset/LastFm_valid.csv";
# matX_train, matX_test, matX_valid, M, N = LoadUtilities(train_path, test_path, valid_path)

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
  M = max(max(matIndex_train[:,1]), max(matIndex_test[:,1]), max(matIndex_valid[:,1]));
  N = max(max(matIndex_train[:,2]), max(matIndex_test[:,2]), max(matIndex_valid[:,2]));

  matX_train = sparse(matIndex_train[:,1], matIndex_train[:,2], matIndex_train[:,3], M, N);
  matX_test = sparse(matIndex_test[:,1], matIndex_test[:,2], matIndex_test[:,3], M, N);
  matX_valid = sparse(matIndex_valid[:,1], matIndex_valid[:,2], matIndex_valid[:,3], M, N);

  return matX_train, matX_test, matX_valid
end

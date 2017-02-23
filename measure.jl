function compute_precNrec(matX_ground_truth::SparseMatrixCSC{Float64,2}, matX_infere::Array{Float64,2}, topK::Int64)
  usr_size = size(matX_ground_truth, 1);
  itm_size = size(matX_ground_truth, 2);
  topK_size = length(topK);
  precision = zeros(usr_size, topK_size);
  recall = zeros(usr_size, topK_size);

  if size(matX_infere, 1) != usr_size
    return false
  end

  K = maximum(topK);

  mat_loc = zeros(Int64, usr_size, topK_size);
  for u = 1:usr_size
    (res, loc) = maxk(matX_infere[u,:], K);
    mat_loc[u,:] = loc;
  end

  [usr_id, topK_rank, item_id] = findnz(mat_loc);
  for i = 1:topK_size
    win_size = usr_size * topK[i];

    # dim(accurate_mask) = usr_size * itm_size
    accurate_mask = sparse(usr_id[1:win_size], item_id[1:win_size], ones(win_size), usr_size, itm_size);
    accurate_mask = (accurate_mask .* vec_label) .> 0;
    num_TP = sum(accurate_mask, 2);
    precision[:,i] = num_TP / topK[i];
    recall[:,i] = num_TP ./ sum(vec_label.>0, 2);
  end
end

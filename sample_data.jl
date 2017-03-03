using StatsBase

function sample_data(M::Int64, N::Int64, usr_batch_size::Int64,
                     matX_train::SparseMatrixCSC{Float64,Int64},
                     usr_zeros::BitArray{1}, itm_zeros::BitArray{1})

  if usr_batch_size == M
    usr_idx = collect(1:M);
    itm_idx = collect(1:N);
    deleteat!(usr_idx, usr_zeros);
    deleteat!(itm_idx, itm_zeros);
    usr_idx_len = length(usr_idx);
    itm_idx_len = length(itm_idx);
  else
    usr_idx = sample(collect(1:M), usr_batch_size, replace=false);
    deleteat!(usr_idx, find(sum(matX_train[usr_idx,:],2)[:] .== 0));
    itm_idx = find(sum(matX_train[usr_idx, :], 1) .> 0);
    usr_idx_len = length(usr_idx);
    itm_idx_len = length(itm_idx);
  end
  return usr_idx, itm_idx, usr_idx_len, itm_idx_len
end

#
#  /// --- Unit test for function: evaluate() --- ///
#
 X =  sparse([5. 4 3 0 0 0 0 0;
              3. 4 5 0 0 0 0 0;
              0  0 0 3 3 4 0 0;
              0  0 0 5 4 5 0 0;
              0  0 0 0 0 0 5 4;
              0  0 0 0 0 0 3 4;
              0  0 0 0 0 0 0 0])
(is, js, vs) = findnz(X[2, [2,3,4,5,6]])
is
js
vs
# usr_zeros = (sum(X, 2) .== 0)[:]
# itm_zeros = (sum(X, 1) .== 0)[:]
# usr_idx, itm_idx, usr_idx_len, itm_idx_len = sample_data(7, 8, 2, X, usr_zeros, itm_zeros)

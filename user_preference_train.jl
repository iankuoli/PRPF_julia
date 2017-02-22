include("Lambert_W.jl")

function user_preference_train(vec_prior_X_u::Array{Float64,1}, vec_predict_X_u::Array{Float64,1}, vec_matX_u::Array{Float64,1},
                               delta::Float64, C::Float64, alpha::Float64)

  solution_xui_xuj = zeros(1, length(vec_prior_X_u));

  #
  # Compute logisitic(\hat{x}_{ui}) for all nonzero x_{ui}
  #
  exp_diff_predict_xij_h = exp(-vec_predict_X_u)';
  partial_1_diff_predict_xij_h = exp_diff_predict_xij_h ./ (1 + exp_diff_predict_xij_h);
  partial_1_diff_predict_xij_h[isnan(partial_1_diff_predict_xij_h)] = 1;
  partial_2_diff_predict_xij_h = -exp_diff_predict_xij_h ./ (1 + exp_diff_predict_xij_h) .^ 2;
  partial_2_diff_predict_xij_h[isnan(partial_2_diff_predict_xij_h)] = 1;

  #
  # find s_{ui} for all i ∈ \mathcal{I}_u
  #
  mat_diff_matX_u = broadcast(-, vec_matX_u', vec_matX_u);
  mat_exp_diff_predictX_u = exp(delta * broadcast(-, vec_predict_X_u', vec_predict_X_u));
  mat_logistic_diff_predictX_u = 1 ./ (1+mat_exp_diff_predictX_u);
  matL_partial_sui = full(C/length(vec_matX_u) * delta * broadcast(-, mat_logistic_diff_predictX_u * (mat_diff_matX_u .!= 0), sum(mat_diff_matX_u .> 0,1)));
  matL_partial_sui = broadcast(+, matL_partial_sui, alpha * partial_1_diff_predict_xij_h);
  (partial_1_diff_predict_xij_L, min_idx) = findmin(abs(matL_partial_sui), 1);
  min_idx = convert(Array{Int} ,ceil(min_idx/size(matL_partial_sui,1))); # row-wise
  #min_idx = mod(min_idx, length(matL_partial_sui)); # col-wise

  partial_1_diff_predict_xij_L = sum(matL_partial_sui .* sparse(min_idx[:], collect(1:size(matL_partial_sui,1))[:], ones(1,size(matL_partial_sui,1))[:], size(matL_partial_sui,1), size(matL_partial_sui,1)),1);

  vec_s = vec_predict_X_u[min_idx];

  matL_partial2_sui = -C/length(vec_matX_u) * delta^2 * (mat_logistic_diff_predictX_u .* mat_exp_diff_predictX_u ./ (1+mat_exp_diff_predictX_u)) * (mat_diff_matX_u .!= 0);

  partial_2_diff_predict_xij_L = sum(matL_partial2_sui .* sparse(min_idx[:], collect(1:length(min_idx))[:], ones(length(min_idx),1)[:], length(min_idx), length(min_idx)), 1);
  partial_2_diff_predict_xij_L = partial_2_diff_predict_xij_L + alpha * partial_2_diff_predict_xij_h[min_idx];

  #
  # Compute function l and h
  #
  l_function_s = partial_2_diff_predict_xij_L';
  h_function_s = (partial_1_diff_predict_xij_L' + (0.5 - vec_s') .* l_function_s) + log(vec_prior_X_u)';

  #
  # Estimate \tilde{x}_{ui} approximately by Lamber W function
  #
  W_tmp = -l_function_s .* exp(h_function_s);
  W_toosmall_mask = W_tmp .<= -1/exp(1);
  W_toolarge_mask = W_tmp .> 10e+30;
  W_mask = (ones(length(W_tmp),1) - W_toosmall_mask - W_toolarge_mask) .> 0;

  vec_lambda = zeros(length(vec_prior_X_u), 2);
  vec_lambda[find(W_mask), :] = broadcast(*, [Lambert_W(W_tmp[W_mask], 0) Lambert_W(W_tmp[W_mask], -1)], -1 ./ l_function_s[W_mask]'');
  vec_lambda[find(W_toolarge_mask),:] = -repmat((h_function_s[W_toolarge_mask])'' ./ l_function_s[W_toolarge_mask]'', 1, 2);

  (v_better, i_better) = findmin(abs(broadcast(-, vec_lambda, vec_s')), 2);
  i_better = convert(Array{Int} ,ceil(i_better/length(vec_lambda))); # row-wise
  #i_better = convert(Array{Int} ,mod(i_better, length(vec_lambda))) # col-wise
  mask_better = sparse(collect(1:length(vec_prior_X_u))[:], i_better[:], ones(length(vec_prior_X_u), 1)[:], length(vec_prior_X_u), 2);
  vec_lambda[isnan(vec_lambda)] = 0;

  solution_xui_xuj = sum(vec_lambda .* mask_better, 2)';
  solution_xui_xuj[isnan(solution_xui_xuj)] = vec_predict_X_u[isnan(solution_xui_xuj)];
  solution_xui_xuj[solution_xui_xuj .== Inf] = 1;

  if any(isnan(solution_xui_xuj))
     fprintf("NaN");
  end

  if any(solution_xui_xuj .== Inf)
     fprintf("Inf");
  end

  if any(solution_xui_xuj .== -Inf)
     fprintf("-Inf");
  end

  return solution_xui_xuj
end

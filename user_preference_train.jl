include("Lambert_W.jl")


function compute_l_and_h(partial_1st::Array{Float64,1}, partial_2nd::Array{Float64,1}, vec_s::Array{Float64,1}, vec_prior_X_u::Array{Float64,1})
  #
  # Compute function l and h
  #
  l_function_s = partial_2nd;
  tmp = (0.5 - vec_s) .* l_function_s;
  tmp2 = log(vec_prior_X_u);

  h_function_s = partial_1st + (0.5 - vec_s) .* l_function_s + log(vec_prior_X_u);

  return l_function_s, h_function_s
end


function estimate_variational_params(l_function_s::Array{Float64,1}, h_function_s::Array{Float64,1},
                                     vec_prior_X_u::Array{Float64,1}, vec_predict_X_u::Array{Float64,1}, vec_s::Array{Float64,1})
  #
  # Estimate \tilde{x}_{ui} approximately by Lamber W function
  #
  W_tmp = -l_function_s .* exp(h_function_s);
  W_toosmall_mask = W_tmp .<= -1/exp(1);
  W_toolarge_mask = W_tmp .> 10e+30;
  W_mask = (ones(length(W_tmp),1) - W_toosmall_mask - W_toolarge_mask) .> 0;

  vec_lambda = zeros(length(vec_prior_X_u), 2);
  vec_lambda[find(W_mask), :] = broadcast(*, [Lambert_W(W_tmp[W_mask], 0) Lambert_W(W_tmp[W_mask], -1)], -1 ./ l_function_s[W_mask]);
  vec_lambda[find(W_toolarge_mask),:] = -repmat((h_function_s[W_toolarge_mask])'' ./ l_function_s[W_toolarge_mask]'', 1, 2);

  (v_better, i_better) = findmin(abs(broadcast(-, vec_lambda, vec_s)), 2);
  i_better = convert(Array{Int} ,ceil(i_better/size(vec_lambda, 1))); # row-wise

  mask_better = sparse(collect(1:length(vec_prior_X_u))[:], i_better[:], ones(length(vec_prior_X_u), 1)[:], length(vec_prior_X_u), 2);
  vec_lambda[isnan(vec_lambda)] = 0;

  solution_xui_xuj = sum(vec_lambda .* mask_better, 2)[:];
  solution_xui_xuj[isnan(solution_xui_xuj)] = vec_predict_X_u[isnan(solution_xui_xuj)];
  solution_xui_xuj[solution_xui_xuj .== Inf] = 1;

  return solution_xui_xuj
end


function user_preference_train_pw(vec_prior_X_u::Array{Float64,1}, vec_predict_X_u::Array{Float64,1}, vec_matX_u::Array{Float64,1},
                                  C::Float64, alpha::Float64, delta::Float64=1.)

  solution_xui_xuj = zeros(1, length(vec_prior_X_u))

  #
  # Compute logisitic(\hat{x}_{ui}) for all nonzero x_{ui}
  #
  exp_diff_predict_xij_h = exp(-vec_predict_X_u)
  partial_1_diff_predict_xij_h = exp_diff_predict_xij_h ./ (1 + exp_diff_predict_xij_h)
  partial_1_diff_predict_xij_h[isnan(partial_1_diff_predict_xij_h)] = 1
  partial_2_diff_predict_xij_h = -exp_diff_predict_xij_h ./ (1 + exp_diff_predict_xij_h) .^ 2
  partial_2_diff_predict_xij_h[isnan(partial_2_diff_predict_xij_h)] = 1

  #
  # find s_{ui} for all i ∈ \mathcal{I}_u
  #
  mat_diff_matX_u = broadcast(-, vec_matX_u, vec_matX_u')
  mat_exp_diff_predictX_u = exp(delta * broadcast(-, vec_predict_X_u, vec_predict_X_u'))
  mat_logistic_diff_predictX_u = 1 ./ (1+mat_exp_diff_predictX_u)
  matL_partial_sui = full(C/length(vec_matX_u) * delta * broadcast(-, mat_logistic_diff_predictX_u * (mat_diff_matX_u .!= 0), sum(mat_diff_matX_u .> 0,1)))
  matL_partial_sui = broadcast(+, matL_partial_sui, alpha * partial_1_diff_predict_xij_h)
  (partial_1_diff_predict_xij_L, min_idx) = findmin(abs(matL_partial_sui), 1)

  #min_idx = convert(Array{Int} ,ceil(min_idx/size(matL_partial_sui,1))); # row-wise
  min_idx = mod(min_idx, length(vec_predict_X_u) - 1) + 1; # col-wise

  #println(min_idx)
  #println(vec_prior_X_u)
  #println(vec_predict_X_u)
  #println(vec_matX_u)

  partial_1_diff_predict_xij_L = sum(matL_partial_sui .* sparse(min_idx[:], collect(1:size(matL_partial_sui,1))[:], ones(1,size(matL_partial_sui,1))[:], size(matL_partial_sui,1), size(matL_partial_sui,1)),1)

  vec_s = vec_predict_X_u[min_idx]

  matL_partial2_sui = -C/length(vec_matX_u) * delta^2 * (mat_logistic_diff_predictX_u .* mat_exp_diff_predictX_u ./ (1+mat_exp_diff_predictX_u)) * (mat_diff_matX_u .!= 0)

  partial_2_diff_predict_xij_L = sum(matL_partial2_sui .* sparse(min_idx[:], collect(1:length(min_idx))[:], ones(length(min_idx),1)[:], length(min_idx), length(min_idx)), 1)
  partial_2_diff_predict_xij_L = partial_2_diff_predict_xij_L + alpha * partial_2_diff_predict_xij_h[min_idx]

  #
  # Compute function l and h
  #
  # l_function_s = partial_2_diff_predict_xij_L';
  # tmp = (0.5 - vec_s') .* l_function_s;
  # tmp2 = log(vec_prior_X_u)'';
  # h_function_s = (partial_1_diff_predict_xij_L' + (0.5 - vec_s') .* l_function_s) + log(vec_prior_X_u)'';
  l_function_s, h_function_s = compute_l_and_h(partial_1_diff_predict_xij_L[:], partial_2_diff_predict_xij_L[:], vec_s[:], vec_prior_X_u[:])

  #
  # Estimate \tilde{x}_{ui} approximately by Lamber W function
  #
  W_tmp = -l_function_s .* exp(h_function_s)
  W_toosmall_mask = W_tmp .<= -1/exp(1)
  W_toolarge_mask = W_tmp .> 10e+30
  W_mask = (ones(length(W_tmp),1) - W_toosmall_mask - W_toolarge_mask) .> 0

  vec_lambda = zeros(length(vec_prior_X_u), 2);
  vec_lambda[find(W_mask), :] = broadcast(*, [Lambert_W(W_tmp[W_mask], 0) Lambert_W(W_tmp[W_mask], -1)], -1 ./ l_function_s[W_mask]'')
  vec_lambda[find(W_toolarge_mask),:] = -repmat((h_function_s[W_toolarge_mask])'' ./ l_function_s[W_toolarge_mask]'', 1, 2)

  (v_better, i_better) = findmin(abs(broadcast(-, vec_lambda, vec_s')), 2)
  i_better = convert(Array{Int} ,ceil(i_better/size(vec_lambda, 1))); # row-wise
  #i_better = convert(Array{Int} ,mod(i_better, length(vec_lambda))) # col-wise

  mask_better = sparse(collect(1:length(vec_prior_X_u))[:], i_better[:], ones(length(vec_prior_X_u), 1)[:], length(vec_prior_X_u), 2)
  vec_lambda[isnan(vec_lambda)] = 0

  solution_xui_xuj = sum(vec_lambda .* mask_better, 2)'
  solution_xui_xuj[isnan(solution_xui_xuj)] = vec_predict_X_u[isnan(solution_xui_xuj)]
  solution_xui_xuj[solution_xui_xuj .== Inf] = 1

  if any(isnan(solution_xui_xuj))
     fprintf("NaN");
  end

  if any(solution_xui_xuj .== Inf)
     fprintf("Inf");
  end

  if any(solution_xui_xuj .== -Inf)
     fprintf("-Inf");
  end

  return solution_xui_xuj[:]
end


function user_preference_train_luce(vec_prior_X_u::Array{Float64,1}, vec_predict_X_u::Array{Float64,1}, vec_matX_u::Array{Float64,1},
                                    C::Float64, alpha::Float64, delta::Float64=1., sigma::String="exp")

  decreasing_index_matX_u = sortperm(vec_matX_u, rev=true)
  num_I_u = length(decreasing_index_matX_u)

  partial_1_diff_f = zeros(Float64, num_I_u)
  partial_2_diff_f = zeros(Float64, num_I_u)

  if sigma == "exp"
    # exponential tranformation
    sort_transform_predX = exp(delta * vec_predict_X_u[decreasing_index_matX_u])

    matL_partial_sui = zeros(Float64, num_I_u, num_I_u)

    vec_sort_transform_predX = zeros(Float64, length(sort_transform_predX))
    for i=1:num_I_u
      vec_sort_transform_predX[i] = sum(sort_transform_predX[i:num_I_u])
    end

    vec_b = 0
    @time for pi_ui  = 1:num_I_u
      #println(pi_ui)
      vec_b = 0

      #vec_b = sum(broadcast(*, 1 ./ broadcast(+, sort_transform_predX', vec_sort_transform_predX[1:pi_ui]) - sort_transform_predX[pi_ui], sort_transform_predX'), 1)[:]
      for j = 1:pi_ui
       #print(j)
       vec_b += sort_transform_predX ./ (sort_transform_predX + (vec_sort_transform_predX[j] - sort_transform_predX[pi_ui]))
                                         #sum(sort_transform_predX[j:(pi_ui-1)]) +
                                         #sum(sort_transform_predX[(pi_ui+1):num_I_u]))
      end
      #println(",")
      matL_partial_sui[pi_ui, :] = 1 - delta * vec_b + alpha ./ (1 + exp(vec_predict_X_u[decreasing_index_matX_u]))
    end

    (partial_1_diff_predict_xij_L, min_idx) = findmin(abs(matL_partial_sui), 2);
    partial_1_diff_f = matL_partial_sui[min_idx];
    min_idx = convert(Array{Int} ,ceil(min_idx/size(matL_partial_sui,1))); # row-wise


    transform_sui = sort_transform_predX[min_idx];
    exp_sui = exp(vec_predict_X_u[decreasing_index_matX_u[min_idx]]);
    for pi_ui  = 1:num_I_u
      sum_b = 0;
      for j = 1:pi_ui
        vec_tmp = transform_sui[pi_ui] ./ (transform_sui[pi_ui] + (vec_sort_transform_predX[j] - sort_transform_predX[pi_ui]))
                                           #sum(sort_transform_predX[j:(pi_ui-1)]) +
                                           #sum(sort_transform_predX[(pi_ui+1):num_I_u]));
        sum_b += vec_tmp * (1 - vec_tmp);
      end
      partial_2_diff_f[pi_ui] = - delta^2 * sum_b - alpha * exp_sui[pi_ui] / (1 + exp_sui[pi_ui])^2;
    end

  else
    # linear transformation
    sort_transform_predX = delta * vec_predict_X_u[decreasing_index_matX_u];

    transform_predX = delta * vec_predict_X_u;

    matL_partial_sui = zeros(Float64, num_I_u, num_I_u);
    @time for pi_ui  = 1:num_I_u
      vec_b = 0;
      for j = 1:pi_ui
        vec_b += sort_transform_predX ./ (sort_transform_predX +
                                          sum(sort_transform_predX[j:(pi_ui-1)]) +
                                          sum(sort_transform_predX[(pi_ui+1):num_I_u]));
      end
      matL_partial_sui[pi_ui, :] = 1 ./ vec_predict_X_u[decreasing_index_matX_u] - delta * vec_b + alpha ./ (1 + exp(vec_predict_X_u[decreasing_index_matX_u]));
    end

    (partial_1_diff_predict_xij_L, min_idx) = findmin(abs(matL_partial_sui), 2);
    partial_1_diff_f = matL_partial_sui[min_idx];
    min_idx = convert(Array{Int} ,ceil(min_idx/size(matL_partial_sui,1))); # row-wise

    transform_sui = sort_transform_predX[min_idx];
    sui = vec_predict_X_u[decreasing_index_matX_u[min_idx]];

    exp_sui = exp(sui);
    for pi_ui  = 1:num_I_u
      sum_b = 0;
      for j = 1:pi_ui
        vec_tmp = transform_sui[pi_ui] ./ (transform_sui[pi_ui] +
                                           sum(sort_transform_predXu[j:(pi_ui-1)]) +
                                           sum(sort_transform_predX[(pi_ui+1):num_I_u]));
        sum_b += vec_tmp * (1 - vec_tmp);
      end
      partial_2_diff_f[pi_ui] = 1/sui[pi_ui]^2 - delta^2 * sum_b - alpha * exp_sui[pi_ui] / (1 + exp_sui[pi_ui])^2;
    end
  end

  #
  # Compute function l and h
  #
  vec_s = vec_predict_X_u[decreasing_index_matX_u[min_idx]];

  l_function_s, h_function_s = compute_l_and_h(partial_1_diff_f[:], partial_2_diff_f[:], vec_s[:], vec_prior_X_u[decreasing_index_matX_u[min_idx]][:]);

  #
  # Estimate \tilde{x}_{ui} approximately by Lamber W function
  #
  solution_xui_xuj = estimate_variational_params(l_function_s, h_function_s, vec_prior_X_u, vec_predict_X_u, vec_s[:]);

  #
  # So far, the indices of partial_1_diff_f and partial_2_diff_f follow decreasing order.
  # Now we will map the sorted index to the original index.
  #
  qqq = full(SparseVector(num_I_u, decreasing_index_matX_u, collect(1:num_I_u)))

  #println(partial_1_diff_f)
  #println(partial_2_diff_f)
  #println(solution_xui_xuj)
  #println(decreasing_index_matX_u)
  #println(qqq)

  solution_xui_xuj = solution_xui_xuj[qqq]

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


#
#  /// --- Unit test for function: evaluate() --- ///
#

# vec_prior_X_u = [2., 5., 7., 8.]
# vec_predict_X_u = [3., 4., 5., 6.]
# vec_matX_u = [2., 4., 10., 20.]

# vec_prior_X_u = [10., 10., 10., 10.]
# vec_predict_X_u = [5.1, 5.2, 5.0, 4.8]
# vec_matX_u = [20., 44., 100., 200.]

vec_prior_X_u = rand(500)
vec_predict_X_u = rand(500)
vec_matX_u = rand(500)


delta = 1.
C=10.
alpha = 1000.
@time user_preference_train_pw(vec_prior_X_u, vec_predict_X_u, vec_matX_u, C, alpha, delta)
user_preference_train_luce(vec_prior_X_u, vec_predict_X_u, vec_matX_u, C, alpha, delta)

vec_predict_X_u







#

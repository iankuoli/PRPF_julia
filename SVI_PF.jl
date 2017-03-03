include("user_preference_train.jl");

#
# tensorPhi_{uik} = ρ_{uik} * x_{ui},
# where ρ_{uik} = ψ(Θ_{uk}) - log(Θ_{uk}) + ψ(β_{ik}) - log(β_{ik})
#
function Update_tensorPhi(predict_X::SparseMatrixCSC{Float64,Int64},
                          matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                          matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2})

  m = size(predict_X, 1);
  n = size(predict_X, 2);
  K = size(matTheta_Shp, 2);

  tensorPhi = spzeros(m, n*K);
  sum_tensorPhi = spzeros(m, n);

  matTheta_Shp_psi = digamma(matTheta_Shp);
  matTheta_Rte_log = log(matTheta_Rte);
  matBeta_Shp_psi = digamma(matBeta_Shp);
  matBeta_Rte_log = log(matBeta_Rte);

  matX_One = predict_X .> 0;
  for k = 1:K
    tensorPhi[:, ((k-1)*n+1):(k*n)] = broadcast(*, matX_One, (matTheta_Shp_psi[:,k] - matTheta_Rte_log[:,k])) +
                                      broadcast(*, matX_One, (matBeta_Shp_psi[:,k] - matBeta_Rte_log[:,k])');
    sum_tensorPhi += tensorPhi[:, ((k-1)*n+1):(k*n)];
  end
  for k = 1:K
    tensorPhi[:, ((k-1)*n+1):(k*n)] = predict_X .* tensorPhi[:, ((k-1)*n+1):(k*n)] ./ sum_tensorPhi;
  end

  return tensorPhi
end


function Update_matTheta(lr::Float64, tensorPhi::SparseMatrixCSC{Float64,Int64},
                         matBeta::Array{Float64,2}, a::Float64, matEpsilon::Array{Float64,1},
                         matTheta::Array{Float64,2}, matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2})
  m = size(matTheta, 1);
  n = size(matBeta, 1);
  K = size(matTheta, 2);

  for k = 1:K
    matTheta_Shp[:, k] = (1 - lr) * matTheta_Shp[:, k] + lr * (a + sum(tensorPhi[:, ((k-1)*n+1):(k*n)], 2));
  end
  matTheta_Rte = (1 - lr) * matTheta_Rte[:, k] + lr * broadcast(+, repmat(sum(matBeta, 1), m, 1), matEpsilon);
  matTheta = matTheta_Shp ./ matTheta_Rte;

  return matTheta, matTheta_Shp, matTheta_Rte
end


function Update_matBeta(lr::Float64, tensorPhi::SparseMatrixCSC{Float64,Int64},
                        matTheta::Array{Float64,2}, d::Float64, matEta::Array{Float64,1},
                        matBeta::Array{Float64,2}, matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2})
  m = size(matTheta, 1);
  n = size(matBeta, 1);
  K = size(matTheta, 2);

  for k = 1:K
    matBeta_Shp[:, k] = (1 - lr) * matBeta_Shp[:, k] + lr * (a + sum(tensorPhi[:, ((k-1)*n+1):(k*n)], 1)');
  end
  matBeta_Rte = (1 - lr) * matBeta_Rte[:, k] + lr * broadcast(+, repmat(sum(matTheta, 1), n, 1), matEta);
  matBeta = matBeta_Shp ./ matBeta_Rte;

  return matBeta, matBeta_Shp, matBeta_Rte
end


function Update_matEpsilon(lr::Float64, matTheta::Array{Float64,2}, a::Float64, b::Float64, c::Float64,
                           matEpsilon::Array{Float64,1}, matEpsilon_Shp::Array{Float64,1}, matEpsilon_Rte::Array{Float64,1})
  K = size(matTheta, 2);
  matEpsilon_Shp = (1-lr) * matEpsilon_Shp + lr * (b + K * a);
  matEpsilon_Rte = (1-lr) * matEpsilon_Rte + lr * (c + sum(matTheta, 2));
  matEpsilon = matEpsilon_Shp ./ matEpsilon_Rte;
  return matEpsilon, matEpsilon_Shp, matEpsilon_Rte
end


function Update_matEta(lr::Float64, matBeta::Array{Float64,2}, d::Float64, e::Float64, f::Float64,
                       matEta::Array{Float64,1}, matEta_Shp::Array{Float64,1}, matEta_Rte::Array{Float64,1})
  K = size(matBeta, 2);
  matEta_Shp = (1-lr) * matEta_Shp + lr * (e + K * d);
  matEta_Rte = (1-lr) * matEta_Rte + lr * (f + sum(matBeta, 2));
  matEta = matEta_Shp ./ matEta_Rte;
  return matEta, matEta_Shp, matEta_Rte
end


function SVI_PF(lr::Float64, M::Int64, N::Int64, K::Int64, ini_scale::Float64, usr_idx::Array{Int64,1}, itm_idx::Array{Int64,1},
                predict_X::SparseMatrixCSC{Float64,Int64},
                matTheta::Array{Float64,2}, matTheta_Shp::Array{Float64,2}, matTheta_Rte::Array{Float64,2},
                matBeta::Array{Float64,2}, matBeta_Shp::Array{Float64,2}, matBeta_Rte::Array{Float64,2},
                matEpsilon::Array{Float64,2}, matEpsilon_Shp::Array{Float64,2}, matEpsilon_Rte::Array{Float64,2},
                matEta::Array{Float64,2}, matEta_Shp::Array{Float64,2}, matEta_Rte::Array{Float64,2})

  m = length(usr_idx);
  n = length(itm_idx);

  #
  # Update tensorPhi
  #
  print("Update tensorPhi ... ");

MethodError: no method matching Update_tensorPhi(::SparseMatrixCSC{Float64,Int64}, ::Array{Int64,2}, ::Array{Int64,2}, ::Array{Int64,2}, ::Array{Int64,2})
Closest candidates are:
  Update_tensorPhi(::SparseMatrixCSC{Float64,Int64}, !Matched::Array{Float64,2}, !Matched::Array{Float64,2}, !Matched::Array{Float64,2}, !Matched::Array{Float64,2}) at /home/ian/workspace/julia/PRPF_julia/SVI_PF.jl:11
 in include_string(::String, ::String) at loading.jl:441
 in include_string(::String, ::String, ::Int64) at eval.jl:28
 in include_string(::Module, ::String, ::String, ::Int64, ::Vararg{Int64,N}) at eval.jl:32
 in (::Atom.##53#56{String,Int64,String})() at eval.jl:40
 in withpath(::Atom.##53#56{String,Int64,String}, ::String) at utils.jl:30
 in withpath(::Function, ::String) at eval.jl:46
 in macro expansion at eval.jl:57 [inlined]
 in (::Atom.##52#55{Dict{String,Any}})() at task.jl:60
  tensorPhi = Update_tensorPhi(predict_X, matTheta_Shp, matTheta_Rte, matBeta_Shp, matBeta_Rte);

  matTheta, matTheta_Shp, matTheta_Rte = Update_matTheta(lr, tensorPhi, matBeta, a, matEpsilon, matTheta, matTheta_Shp, matTheta_Rte);
  matBeta, matBeta_Shp, matBeta_Rte = Update_matBeta(lr, tensorPhi, matTheta, d, matEta, matBeta, matBeta_Shp, matBeta_Rte);
  matEpsilon, matEpsilon_Shp, matEpsilon_Rte = Update_matEpsilon(lr, matTheta, a, b, c, matEpsilon, matEpsilon_Shp, matEpsilon_Rte);
  matEta, matEta_Shp, matEta_Rte = Update_matEta(lr, matBeta, d, e, f, matEta, matEta_Shp, matEta_Rte);

  return matTheta, matTheta_Shp, matTheta_Rte, matBeta, matBeta_Shp, matBeta_Rte,
         matEpsilon, matEpsilon_Shp, matEpsilon_Rte, matEta, matEta_Shp, matEta_Rte
  #
  #
  # matTheta[usr_idx,:], matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:] = Update_matTheta(lr, tensorPhi, matBeta[itm_idx,:], a, matEpsilon[usr_idx],
  #                                                                                         matTheta[usr_idx,:], matTheta_Shp[usr_idx,:], matTheta_Rte[usr_idx,:]);
  #
  # matBeta[itm_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:] = Update_matBeta(lr, tensorPhi, matTheta[usr_idx,:], d, matEta[itm_idx],
  #                                                                                     matBeta[itm_idx,:], matBeta_Shp[itm_idx,:], matBeta_Rte[itm_idx,:]);
  #
  # matEpsilon[usr_idx], matEpsilon_Shp[usr_idx], matEpsilon_Rte[usr_idx] = Update_matEpsilon(lr, matTheta[usr_idx,:], a, b, c,
  #                                                                                           matEpsilon[usr_idx], matEpsilon_Shp[usr_idx], matEpsilon_Rte[usr_idx]);
  #
  # matEta[usr_idx], matEta_Shp[usr_idx], matEta_Rte[usr_idx] = Update_matEta(lr, matBeta[itm_idx,:], d, e, f,
  #                                                                           matEta[itm_idx], matEta_Shp[itm_idx], matEta_Rte[itm_idx]);
end


predict_X =  sparse([5. 4 3 0 0 0 0 0;
                     3. 4 5 0 0 0 0 0;
                     0  0 0 3 3 4 0 0;
                     0  0 0 5 4 5 0 0;
                     0  0 0 0 0 0 5 4;
                     0  0 0 0 0 0 3 4;
                     0  0 0 0 0 0 0 0])
matTheta_Shp = [1. 0 0; 1 0 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 0]+10e-10
matTheta_Rte = [1. 0 0; 1 0 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 0]+10e-10
matBeta_Shp = [4. 0 0; 4 0 0; 4 0 0; 0 4 0; 0 3 0; 0 5 0; 0 0 4; 0 0 4]+10e-10
matBeta_Rte = [4. 0 0; 4 0 0; 4 0 0; 0 4 0; 0 3 0; 0 5 0; 0 0 4; 0 0 4]+10e-10
T = Update_tensorPhi(X, matTheta_shp, matTheta_rte, matBeta_shp, matBeta_rte)

m = size(predict_X, 1);
n = size(predict_X, 2);
K = size(matTheta_Shp, 2);

tensorPhi = spzeros(m, n*K);
sum_tensorPhi = spzeros(m, n);

matTheta_Shp_psi = digamma(matTheta_Shp)
matTheta_Rte_log = log(matTheta_Rte)
matBeta_Shp_psi = digamma(matBeta_Shp)
matBeta_Rte_log = log(matBeta_Rte)

matX_One = predict_X .> 0

(matTheta_Shp_psi[:,1] - matTheta_Rte_log[:,1])''
broadcast(*, matX_One, (matTheta_Shp_psi[:,1] - matTheta_Rte_log[:,1])'')- broadcast(*, matX_One, (matBeta_Shp_psi[:,1] - matBeta_Rte_log[:,1])')

for k = 1:K
  tensorPhi[:, ((k-1)*n+1):(k*n)] = broadcast(*, matX_One, (matTheta_Shp_psi[:,k] - matTheta_Rte_log[:,k])) +
                                             broadcast(*, matX_One, (matBeta_Shp_psi[:,k] - matBeta_Rte_log[:,k])');
  sum_tensorPhi += tensorPhi[:, ((k-1)*n+1):(k*n)];
end
tensorPhi
sum_tensorPhi
for k = 1:K
  tensorPhi[:, ((k-1)*n+1):(k*n)] = predict_X .* tensorPhi[:, ((k-1)*n+1):(k*n)] ./ sum_tensorPhi;
end
tensorPhi

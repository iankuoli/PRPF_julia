include("maxK.jl")

function inference(usr_idx::Array{Int64,1}, matTheta::Array{Float64,2}, matBeta::Array{Float64,2})
  return matTheta[usr_idx,:] * matBeta';
end


function infer_entry(matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, i_idx, j_idx)
  return (matTheta[i_idx,:]' * matBeta[j_idx, :])[1]
end

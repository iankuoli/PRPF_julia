using HDF5

function write_model(file_name, matTheta, matBeta, matEpsilon, matEta, prior, C, delta, alpha)
  path = string("PRPF_julia/model/", file_name, ".h5")

  if isfile(path)
    rm(path)
  end

  print(alpha)

  h5write(path, "PRPF/matTheta", matTheta)
  h5write(path, "PRPF/matBeta", matBeta)
  h5write(path, "PRPF/matEpsilon", matEpsilon)
  h5write(path, "PRPF/matEta", matEta)
  h5write(path, "PRPF/prior", collect(prior))
  h5write(path, "PRPF/C", C)
  h5write(path, "PRPF/delta", delta)
  h5write(path, "PRPF/alpha", alpha)
end

function read_model(file_name)
  path = string("PRPF_julia/model/", file_name, ".h5")
  matTheta = h5read(path, "PRPF/matTheta")
  matBeta = h5read(path, "PRPF/matBeta")
  matEpsilon = h5read(path, "PRPF/matEpsilon")
  matEta = h5read(path, "PRPF/matEta")
  prior = tuple(h5read(path, "PRPF/prior")...)
  C = h5read(path, "PRPF/C")
  delta = h5read(path, "PRPF/delta")
  alpha = h5read(path, "PRPF/alpha")

  return matTheta, matBeta, matEpsilon, matEta, prior, C, delta, alpha
end

function maxK(vec::Array{Float64,1}, K::Int64)
  retVal = zeros(Float64, K);
  retIdx = zeros(Int64, K);

  heap_v = Base.Collections.heapify(vec, Base.Order.Reverse);

  k = K;
  while k > 0
    topk_val = Base.Collections.heappop!(heap_v, Base.Order.Reverse);
    topk_idx = findn(vec.==topk_val);
    println(topk_idx);
    retVal[K-k+1] = topk_val;
    retIdx[K-k+1] = topk_idx[1];
    k -= 1;

    if length(topk_idx)>1
      for kk = 1:min(k, length(topk_idx)-1)
        Base.Collections.heappop!(heap_v, Base.Order.Reverse);
        retVal[K-k+1] = topk_val;
        retIdx[K-k+1] = topk_idx[kk+1];
        k -= 1;
      end
    end
  end
  return retVal, retIdx
end

#
#  /// --- Unit test --- ///
#
#a = [5.,6.,3.,1.,2.,5.,2.,3.]
#maxK(a, 5)

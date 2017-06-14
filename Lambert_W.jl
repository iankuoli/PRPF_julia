function Lambert_W(x, branch = 0, max_itr = 1000)
  # Lambert_W  Functional inverse of x = w*exp(w).
  # w = Lambert_W(x), same as Lambert_W(x,0)
  # w = Lambert_W(x,0)  Primary or upper branch, W_0(x)
  # w = Lambert_W(x,-1)  Lower branch, W_{-1}(x)
  #
  # See: http://blogs.mathworks.com/cleve/2013/09/02/the-lambert-w-function/

  # Copyright 2013 The MathWorks, Inc.

  # Effective starting guess
  if branch == 0
    #  Start above -1
    w = ones(length(x));
  else
    # Start below -1
    w = -2 * ones(length(x));
  end
  v = Inf * w;

  # Haley's method
  i = 0;
  while any(abs.(w - v)./abs.(w) .> 1.e-2)
     i = i + 1;
     v = w;
     e = exp.(w);
     f = w .* e - x;  # Iterate to make this quantity zero
     w = w - f./((e.*(w+1) - (w+2).*f./(2*w+2)));

     if i > max_itr
         #fprintf('not converge');
         return ones(Float64, length(x));
     end
   end
   return w;
end

#
# // ----- Unit Test ----- //
#
# a = Lambert_W([150, 1500])
# a[1]

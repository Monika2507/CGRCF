function channel_weight=ADMM_solve_q(mu,beta,T,h_f,l_f,g_f,channel_weight,L)
     h_f=h_f(:,:,1:31);
     g_f=g_f(:,:,1:31);
     l_f=l_f(:,:,1:31);
     lhd= T ./  (h_f .^2 *mu*T - beta); 
     X=ifft2(mu*g_f + l_f);
     channel_weight=bsxfun(@times,lhd,X);

end
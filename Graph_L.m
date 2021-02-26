function L=Graph_L(xf,state)

     xf=xf(:,:,1:31);
     x=ifft2(xf);
     [r,c,d]=size(x);
     W=(single(zeros([r c d d])));
     sigma_i=10; sigma_j=10;
     for i=1:d
         W(:,:,:,i)=repmat(x(:,:,i),[1 1 d]);
     end
     x_new=repmat(x,[1 1 1 d]);
     W=W-x_new;
     W=W.^2;
     W=sum(sum(W,2),1);
     W=reshape(W,[d d]);
     W=W./(sigma_i*sigma_j);
     L=diag(sum(W,2))-W;
  
      vnm=[num2str(state.frame),'.jpg'];
      Lr=real(W);
     Lr = Lr.*~eye(size(Lr));
end
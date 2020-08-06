function [X] = Chan3D(BSN,BS,R)

%% ��һ��WLS

 %k=X^2+Y^2+Z^2
for i = 1:BSN                      %BSNΪ��վ����
    k(1,i) = BS(1,i)^2 + BS(2,i)^2 + BS(3,i)^2;  %BSΪ��վ����
end

%h = 1/2(Ri^2-ki+k1)
for i =1:BSN-1
    h(i,1) = 0.5*(R(i)^2 - k(1,i+1) + k(1,1));  %ע��k(i+1)
end


%Ga = [Xi,Yi,Zi,Ri]
for i = 1:BSN-1
    Ga(i,1) = -BS(1,i+1);
    Ga(i,2) = -BS(2,i+1);
    Ga(i,3) = -BS(3,i+1);
    Ga(i,4) = -R(i);
end

%QΪTDOAϵͳ��Э�������
Q = cov(R);

%MS��BS����Ͻ�ʱ
za = pinv(Ga' * inv(Q) * Ga) * Ga' * inv(Q) * h

%% �ڶ���WLS
%h'
X1 = BS(1,1);
Y1 = BS(2,1);
Z1 = BS(3,1);
h2 = [
    (za(1,1) - X1)^2;
    (za(2,1) - Y1)^2;
    (za(3,1) - Z1)^2;
     za(4,1)^2
      ];

%Ga'
Ga2 = [
    1,0,0;
    0,1,0;
    0,0,1;
    1,1,1
    ];

%B'
B2 = [
      za(1,1)-X1,0,0,0;
      0,za(2,1)-Y1,0,0;
      0,0,za(3,1)-Z1,0;
      0,0,0,za(4,1)
      ];
  
%za',�����Զʱ
za2 = pinv( Ga2' * inv(B2) * Ga' * inv(Q) * Ga * inv(B2) * Ga2) * (Ga2' * inv(B2) * Ga' * inv(Q) * Ga * inv(B2)) * h2;

zp(1,1) = abs(za2(1,1))^0.5 + X1;
zp(2,1) = abs(za2(2,1))^0.5 + Y1;
zp(3,1) = abs(za2(3,1))^0.5 + Z1;

X = zp;

end

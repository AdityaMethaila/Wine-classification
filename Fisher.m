
TrainData = train(:,2:end);
TrainLabels= train(:,1);
TestData= test(:,2:end);
TestLabels=test(:,1);
j=0;
k=0;
l=0;
for i=1:90
    if(TrainLabels(i,1)==1)
        j=j+1;
        cluster1(j,:)=TrainData(i,:);
       
        
    elseif(TrainLabels(i,1)==2)
        k=k+1;
        cluster2(k,:)=TrainData(i,:);
        
        
    elseif(TrainLabels(i,1)==3)
        l=l+1;
        cluster3(l,:)=TrainData(i,:);
        
        
    end
end
 
mu1=mean(cluster1);
mu2=mean(cluster2);
mu3=mean(cluster3);
globalmu=mean(TrainData);
d1=cluster1 - repmat(mu1,j,1);
d2=cluster2 - repmat(mu2,k,1);
d3=cluster3 - repmat(mu3,l,1);

covar1=d1'* d1;
covar2=d2'*d2;
covar3=d3'*d3;
Sw=covar1+covar2+covar3;

Sb1= j*(mu1-globalmu)' *(mu1-globalmu);
Sb2 = k*(mu2-globalmu)' *(mu2-globalmu);
Sb3 = l*(mu3-globalmu)' *(mu3-globalmu);
Sb=Sb1+Sb2 + Sb3;

v= inv(Sw) * Sb;
[vec,val]=eig(v);
%%sort v that is eigen vectors and select eigen vectors associated with top
%%eigen values
eigval=diag(val);
[sort_val,sort_val_index] = sort(eigval,'descend');
w=v(:,sort_val_index(1:2));

Y1=TrainData*w;
Y2=TestData*w;

%%%now after the data is reduced to 2 dimensions we will solve it as a
%%%least square method
o1=ones(90,1);
o2=ones(88,1);
TrainData1 = [o1,Y1];
TestData1= [o2,Y2];
for i = 1:90
    if(TrainLabels(i,1)== 1)
        T(i,:)=[1,0,0];
    elseif(TrainLabels(i,1)==2)
        T(i,:) = [0,1,0];
    elseif(TrainLabels(i,1)==3)
        T(i,:) = [0,0,1];
    end
end
W = (inv(TrainData1'*TrainData1)) * (TrainData1'* T);
R = TestData1*W ;
R2 = TrainData1*W

[M,TestY]= max(R,[],2);
[M2,TrainY] = max(R2,[],2)

%%Confusion matrix code starting below%%

for r=1:3
    for c=1:3
        Trainconf(r,c)=0;
        Testconf(r,c)=0;
    end
end

for i=1:90
        a=TrainY(i,1);
        b=TrainLabels(i,1);
        
        Trainconf(a,b)= Trainconf(a,b) + 1;
        
end

for i=1:88
    c=TestY(i,1);
    d=TestLabels(i,1);
    Testconf(c,d) = Testconf(c,d)+1;
end
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

%% now we apply k-means on this projected data 


o1=ones(90,1);
o2=ones(88,1);
TrainData1 = [o1,Y1];
TestData1= [o2,Y2];
TrainLabels= train(:,1);
TestLabels=test(:,1);

index = randperm(90);
for i=1:3
    Mu(i,:)=TrainData1(index(i),:);
end

while(1)
    for i=1:90
        t1= TrainData1(i,:) - Mu(1,:);
        dist(i,1) = sqrt(t1*t1');
        t2= TrainData1(i,:) - Mu(2,:);
        dist(i,2) = sqrt(t2*t2');
        t3= TrainData1(i,:) - Mu(3,:);
        dist(i,3) = sqrt(t3*t3');
    end
    
    j=1;
    k=1;
    l=1;
    
    for i =1:90
        mindist = min([dist(i,1),dist(i,2),dist(i,3)])
        if(mindist==dist(i,1))
            Z(i,1)=1;
            Z(i,2)=0;
            Z(i,3)=0;
            Cluster1(j,:)= TrainData1(i,:);
            j=j+1;
            
        elseif(mindist==dist(i,2))
            Z(i,1)=0;
            Z(i,2)=1;
            Z(i,3)=0;
            Cluster2(k,:)= TrainData1(i,:);
            k=k+1;
        elseif(mindist==dist(i,3))
            Z(i,1)=0;
            Z(i,2)=0;
            Z(i,3)=1;
            Cluster3(l,:)=TrainData1(i,:);
            l=l+1;
            
            
        end
    end 
    
    NewMu(1,:)= mean(Cluster1);
    NewMu(2,:)= mean(Cluster2);
    NewMu(3,:)= mean(Cluster3);
    
    if(Mu==NewMu)
        break;
        
    else
        Mu=NewMu;
        
    end
    
end

% now run the model on test data
for i=1:88
        t1= TestData1(i,:) - Mu(1,:);
        Testdist(i,1) = sqrt(t1*t1');
        t2= TestData1(i,:) - Mu(2,:);
        Testdist(i,2) = sqrt(t2*t2');
        t3= TestData1(i,:) - Mu(3,:);
        Testdist(i,3) = sqrt(t3*t3');
end


%Y3 gives the cluster too which a data point belongs%
[M2 Y3] = min(Testdist,[],2);

%%%%%Confusion matrix prep code%%%%%

%create Z for testing data
for i = 1:88
    if(TestLabels(i,1)== 1)
        TestZ(i,:)=[1,0,0];
    elseif(TestLabels(i,1)==2)
        TestZ(i,:) = [0,1,0];
    elseif(TestLabels(i,1)==3)
        TestZ(i,:) = [0,0,1];
    end
end
%Create Z for training data
for i = 1:90
    if(TrainLabels(i,1)== 1)
        TrainZ(i,:)=[1,0,0];
    elseif(TrainLabels(i,1)==2)
        TrainZ(i,:) = [0,1,0];
    elseif(TrainLabels(i,1)==3)
        TrainZ(i,:) = [0,0,1];
    end
end

j=1;
k=1;
for r=1:3
    for c=1:3
        Trainconf(r,c)=0;
        Testconf(r,c)=0;
    end
end

%creating two confusion matrices for training and testing

for i=1:88
    for j=1:3
        for k =1:3
            if(TestZ(i,j)==1&&Z(i,k)==1)
                Testconf(j,k)=Testconf(j,k)+1
                
            end
        end
    end
end
   

for i=1:90
    for j=1:3
        for k =1:3
            if(TrainZ(i,j)==1&&Z(i,k)==1)
                Trainconf(j,k)=Trainconf(j,k)+1
                
            end
        end
    end
end

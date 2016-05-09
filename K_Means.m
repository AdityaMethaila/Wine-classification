
TrainData=train(:,2:end);
TrainLabels= train(:,1);
TestData= test(:,2:end);
TestLabels=test(:,1);

index = randperm(90);
for i=1:3
    Mu(i,:)=TrainData(index(i),:);
end

while(1)
    for i=1:90
        t1= TrainData(i,:) - Mu(1,:);
        dist(i,1) = sqrt(t1*t1');
        t2= TrainData(i,:) - Mu(2,:);
        dist(i,2) = sqrt(t2*t2');
        t3= TrainData(i,:) - Mu(3,:);
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
            Cluster1(j,:)= TrainData(i,:);
            j=j+1;
            
        elseif(mindist==dist(i,2))
            Z(i,1)=0;
            Z(i,2)=1;
            Z(i,3)=0;
            Cluster2(k,:)= TrainData(i,:);
            k=k+1;
        elseif(mindist==dist(i,3))
            Z(i,1)=0;
            Z(i,2)=0;
            Z(i,3)=1;
            Cluster3(l,:)=TrainData(i,:);
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
        t1= TestData(i,:) - Mu(1,:);
        Testdist(i,1) = sqrt(t1*t1');
        t2= TestData(i,:) - Mu(2,:);
        Testdist(i,2) = sqrt(t2*t2');
        t3= TestData(i,:) - Mu(3,:);
        Testdist(i,3) = sqrt(t3*t3');
end


%Y2 gives the cluster too which a data point belongs%
[M2 Y2] = min(Testdist,[],2);

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

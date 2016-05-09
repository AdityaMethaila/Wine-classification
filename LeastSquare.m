
TrainData = train(:,2:end);
TrainLabels= train(:,1);
TestData= test(:,2:end);
TestLabels=test(:,1);
for i = 1:90
    if(TrainLabels(i,1)== 1)
        T(i,:)=[1,0,0];
    elseif(TrainLabels(i,1)==2)
        T(i,:) = [0,1,0];
    elseif(TrainLabels(i,1)==3)
        T(i,:) = [0,0,1];
    end
end
W = (inv(TrainData'*TrainData)) * (TrainData'* T);
R = TestData*W ;
R2 = TrainData*W

[M,TestY]= max(R,[],2);
[M2,TrainY] = max(R2,[],2)

plot(TrainData);

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





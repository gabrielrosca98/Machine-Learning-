load ORLfacedata
yFinal = zeros(280, 40);
for index = 1:50
    [Xtr, Xte, Ytr, Yte] = PartitionData(data, labels,3);
    dataStructure(index).Xtr = Xtr;
    dataStructure(index).Xte = Xte;
    dataStructure(index).Ytr = Ytr;
    dataStructure(index).Yte = Yte;
end

for indexSubject = 1:40
  indexStart = (indexSubject - 1) * 3 + 1;
  indexFinish = (indexSubject - 1) * 3 + 3;
  XtildaTraining = [ones(3, 1), dataStructure(1).Xtr([indexStart:indexFinish],:)];
  wForSubject(indexSubject).w =  pinv(XtildaTraining) * dataStructure(1).Ytr([indexStart:indexFinish],:);
end

for indexDataSet = 1:50
  for indexSample = 1:280
      minDifference = intmax;
      for indexClassifier = 1:40
        XtildaTesting = [1, dataStructure(indexDataSet).Xte([indexSample],:)];
        predictedClass(indexClassifier) = XtildaTesting * wForSubject(indexClassifier).w;
        difference = abs(indexClassifier - predictedClass(indexClassifier));
        %fprintf('The predicted y:\t%f for sample %d and class %d and indexClassifier%d \t diff:%fminDiff%f\n', predictedClass(indexClassifier), indexSample, dataStructure(indexDataSet).Yte(indexSample), indexClassifier, difference, minDifference)
        if difference < minDifference
          minDifference = difference;
          finalClass = indexClassifier;
        end
        %fprintf('MinDif:%f\t Class:%d\t Sample:%d\t Diff:%f\t FinalCls:%d\n',minDifference, dataStructure(indexDataSet).Yte(indexSample), indexSample,difference, finalClass)
      end
      yFinal(indexSample, finalClass) = 1;
  end
  wrong = 0;
  for indexMatrixYSample = 1:280
    for indexMatrixYClass = 1:40
      if yFinal(indexMatrixYSample, indexMatrixYClass) == 1
        if indexMatrixYClass ~= dataStructure(indexDataSet).Yte(indexMatrixYSample)
          wrong = wrong + 1;
        end
      end
    end
  end
  errorRate(indexDataSet) = (wrong / 280) * 100;
  accuracyRate(indexDataSet) = 100 - errorRate(indexDataSet);
  fprintf('The accuracy rate for data sample %d is %f\n', indexDataSet, accuracyRate(indexDataSet))
  yFinal = zeros(280, 40);
end

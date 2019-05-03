clc
load ORLfacedata
yFinal = zeros(200, 40);
for index = 1:50
    [Xtr, Xte, Ytr, Yte] = PartitionData(data, labels,5);
    dataStructure(index).Xtr = Xtr;
    dataStructure(index).Xte = Xte;
    dataStructure(index).Ytr = Ytr;
    dataStructure(index).Yte = Yte;
end

yClass = zeros(200, 40);
for index = 1:40
  startIndex = (index - 1) * 5 + 1;
  finishIndex = (index - 1) * 5 + 5;
  for indexSubj = startIndex:finishIndex
    yClass(indexSubj, dataStructure(10).Ytr(indexSubj)) = 1;
  end
end

XtildaTraining = [ones(200, 1), dataStructure(10).Xtr];
w = pinv(XtildaTraining) * yClass;
w = transpose(w);

for indexDataSet = 1:50
  wrongClass = 0;
  for indexSample = 1:200
    maximum = intmin;
    finalClass = 0;
    for indexClassifier = 1:40
      Xtesting = [1, dataStructure(indexDataSet).Xte(indexSample,:)];
      yResult(indexSample,indexClassifier) = Xtesting * transpose(w(indexClassifier,:));
      if yResult(indexSample, indexClassifier) > maximum
        maximum = yResult(indexSample, indexClassifier);
        finalClass = indexClassifier;
      end
    end
    if finalClass ~= dataStructure(indexDataSet).Yte(indexSample)
      wrongClass = wrongClass + 1;
    end
  end
  accRateLinear(indexDataSet) = 100 - (wrongClass / 200) * 100;
  fprintf('The acc for data set %d is %f\n', indexDataSet, accRateLinear(indexDataSet))
end

for indexDataSet = 1:20
  for indexSubject = 1:40
    wrongSubject = 0;
    maximum = intmin;
    finalClass = 0;
    for indexSample = 1:5
      Xtesting = [1, dataStructure(indexDataSet).Xte((indexSubject - 1) * 5 + indexSample,:)];
      for indexClassifier = 1:40
        ySubjTesting =  Xtesting * transpose(w(indexClassifier, :));
        if ySubjTesting > maximum
          maximum = ySubjTesting;
          finalClass = indexClassifier;
        end
      end
      if finalClass ~= dataStructure(indexDataSet).Yte((indexSubject - 1) * 5 + indexSample)
        wrongSubject = wrongSubject + 1;
      end
    end
    accSubjLinear(indexDataSet, indexSubject) = 100 - (wrongSubject/ 5) * 100;
  end
end

indexDataSet = 1:50;
figure('Name', 'Linear multi-class vs kNN')
plot(indexDataSet, accRateLinear, indexDataSet,accuracyTesting);
xlabel('Index data set');
ylabel('Accuracy testing');

indexSubject = 1:40;
meanSubject = mean(accSubjLinear);
plot(indexSubject, meanSubject)
xlabel('Index subject')
ylabel('Mean accuracy')

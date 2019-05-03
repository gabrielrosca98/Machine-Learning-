function [valuesFeatures, uniqueValuesParam] = NBTrain(AttributeSet, LabelSet)
  %Get the unique values for classes
  numberClasses = size(unique(LabelSet), 1);
  uniqueClasses = unique(LabelSet);
  uniqueClasses = [uniqueClasses, zeros(size(uniqueClasses, 1) ,1)];

  %Determine how many unique values for the features we have
  for indexFeature = 1:size(AttributeSet, 2)
    uniqueValuesParam(indexFeature).Matrix = unique(AttributeSet(:, indexFeature));
    uniqueValuesParam(indexFeature).Matrix = [uniqueValuesParam(indexFeature).Matrix, zeros(size(uniqueValuesParam(indexFeature).Matrix, 1), 1)];
    uniqueValuesParam(indexFeature).ClassesFrequency = [uniqueClasses, zeros(size(uniqueClasses, 1), size(uniqueValuesParam(indexFeature).Matrix, 1))];
  end
  %Count the frequency for each feature
  for indexSample = 1:size(AttributeSet, 1)
    for indexFeature = 1:size(AttributeSet, 2)
      featureRowIndex = find(uniqueValuesParam(indexFeature).Matrix(:, 1) == AttributeSet(indexSample,indexFeature));
      uniqueValuesParam(indexFeature).Matrix(featureRowIndex,2) = uniqueValuesParam(indexFeature).Matrix(featureRowIndex,2) + 1;
      for indexClass = 1:numberClasses
        uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, featureRowIndex + 1) = uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, featureRowIndex + 1) + sum(uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, 1) ==  LabelSet(indexSample) & AttributeSet(indexSample, indexFeature) == uniqueValuesParam(indexFeature).Matrix(featureRowIndex, 1));
      end
    end
  end
  %Get a vector to hold each value of a feature for a particular class
  valuesFeatures(size(AttributeSet, 2), numberClasses).Vector = [];
  for indexFeature = 1:size(AttributeSet, 2)
    for indexSample = 1:size(AttributeSet, 1)
      valuesFeatures(indexFeature, find(uniqueClasses(:, 1) == LabelSet(indexSample))).Vector = [valuesFeatures(indexFeature, find(uniqueClasses(:, 1) == LabelSet(indexSample))).Vector, AttributeSet(indexSample, indexFeature)];
    end
  end
  %Calculate the mean and std
  for indexFeature = 1:size(AttributeSet, 2)
    for indexSample = 1:size(AttributeSet, 1)
      valuesFeatures(indexFeature, find(uniqueClasses(:, 1) == LabelSet(indexSample))).Mean = mean(valuesFeatures(indexFeature, find(uniqueClasses(:, 1) == LabelSet(indexSample))).Vector);
      valuesFeatures(indexFeature, find(uniqueClasses(:, 1) == LabelSet(indexSample))).Std = std(valuesFeatures(indexFeature, find(uniqueClasses(:, 1) == LabelSet(indexSample))).Vector);
    end
  end
end

function [probabilityModel, uniqueValuesParam] = NBTrain(AttributeSet, LabelSet)
  %Get the unique values for classes
  numberClasses = size(unique(LabelSet), 1);
  uniqueClasses = unique(LabelSet);
  uniqueClasses = [uniqueClasses, zeros(size(uniqueClasses, 1) ,1)];
  %Initialise the structures
  for indexFeature = 1:size(AttributeSet, 2)
    uniqueValuesParam(indexFeature).Matrix = unique(AttributeSet(:, indexFeature));
    uniqueValuesParam(indexFeature).Matrix = [uniqueValuesParam(indexFeature).Matrix, zeros(size(uniqueValuesParam(indexFeature).Matrix, 1), 1)];
    uniqueValuesParam(indexFeature).ClassesFrequency = [uniqueClasses, zeros(size(uniqueClasses, 1), size(uniqueValuesParam(indexFeature).Matrix, 1) - 1)];
  end
  %Determine how many unique values for the features we have
  for indexSample = 1:size(AttributeSet, 1)
    for indexFeature = 1:size(AttributeSet, 2)
      featureRowIndex = find(uniqueValuesParam(indexFeature).Matrix(:, 1) == AttributeSet(indexSample,indexFeature));
      uniqueValuesParam(indexFeature).Matrix(featureRowIndex, 2) = uniqueValuesParam(indexFeature).Matrix(featureRowIndex,2) + 1;
      for indexClass = 1:numberClasses
        uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, featureRowIndex + 1) = uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, featureRowIndex + 1) + sum(uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, 1) ==  LabelSet(indexSample) & AttributeSet(indexSample, indexFeature) == uniqueValuesParam(indexFeature).Matrix(featureRowIndex, 1));
      end
    end
  end
  %Calculate the probability model
  for indexFeature = 1:size(AttributeSet, 2)
    for indexClass = 1:numberClasses
      suma(indexFeature, indexClass) = sum(LabelSet == uniqueClasses(indexClass));
      for indexUniqueValueFeature = 1:size(uniqueValuesParam(indexFeature).Matrix, 1)
        probabilityModel(indexFeature, indexClass).ValueProbability(indexUniqueValueFeature, 1) = (uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, indexUniqueValueFeature + 1) / suma(indexFeature, indexClass)) * 100;
      end
    end
  end
end

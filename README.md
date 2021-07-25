# MFE: A Multi-task Learning Framework for Highly Imbalanced Overlapped Data Classification

Codes for the paper  ‘*MFE: A Multi-task Learning Framework for Highly Imbalanced Overlapped Data Classification*’, by Huiran Yan, Zenghao Cui and Rui Wang.

## Requirements

- python== 3.6
- tensorflow==1.9.0 

## Training

- Step 1:  update your own SavePrefix  path and dataprefix in the "first_Stage_main.py" and  "second_Stage_main.py".
- Step 2:   Set the "nowDim" in "first_Stage_Params.py" and "second_Stage_Params.py"  as your own feature dimension of your dataset.
- Step 3:  Run the "first_Stage_main.py" to choose a best model.  If your want to simply use MFE in  a multi-task way. Classifier in this model can be directly used to do predictions.
- Step 4: If your want to use the MFE as a pretext task to train the feature extractor,  you can set the "lastmodelname" in "second_Stage_Params.py" as the filename of the saved model in Step3. And you can run the  "second_Stage_main.py"  to  only train a new DN (you can design a deeper DN) in this step.

## Notification 

   We also leave one processed Credit Fraud Dataset in "./creditFraud/" (only with one-hot and standScale preprocessing steps), we use 60% of the full data as the training set and 20% as the verification set, which can be used by neural network for early stopping, and the rest 20% as the test set. 

  If you want to simply test the code, you needn't to change the params in the code. You just need to update the dataPrefix and SavePrefix path, and you can run the code following the Step3 and Step4 mentioned above.


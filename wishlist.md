
# Wishlist

 - image3D models
 - tSNE plotting https://www.kaggle.com/philschmidt/quora-eda-model-selection-roc-pr-plots
 - class_overlap - multilabel classification 
 - predictions/ 
   - analyze_preds.py   - ground truth known
     - generateMetrics
     - generateStatistics 
   - compareModel.py - takes list of preds_filenames, if None - provide list of predictions - ground truth unknown
     - compareModelPred  - metrics, statistics
     - compareModelPred_Stratified - stratified comparison
     - scatterplot
     - train_test_deviation
     - generate_report 
   - ensemble.py
              average
              weighted
              find_weights
              NN_based
              Stacking
 - metrics/
        classificationMetrics
        regressionMetrics
        segmentationMetrics

        classificationLoss
        regressionLoss
        segmentationLoss
 
 - Utils/ 
   - featureSelection.py
     - Automated Feature Engineering Basics  - https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics?scriptVersionId=4120225
     - https://www.featuretools.com/
   - DownloadData.py
     - fromKaggle, fromGithub, fromDrive
     - download_model_weights.py
   - dataModifier.py
     - Kfold_generator - https://www.kaggle.com/satishgunjal/tutorial-k-fold-cross-validation#Introduction-
     - Startified_CV_generator - https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
   - dicomHelper - dicomWindowding, DICOM_to_PNG , metadataExtractor

   - explainability.py
     - Shap

   - dimensionReduction.py 
     - tSNE - Image, tabular data 
     - PCA - Image, tabular data
     - KNN clustering - Images, data
   
   - medical_preproces.py
     - segment_lungs
     - image_registration
     - Images.py - nifti, tif --> png files
   - augmentations.py 
     - custom augmentations
   - CalculatePixelInfo  - rgb( mean and std) for your dataset



##  Model types
### tabular models
type = 'lightgbm'

### CNN models
problem =  "segmentation"




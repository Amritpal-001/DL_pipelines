
for any function documentaion - use
 - help(function_name)

ImageDataloader


/src/
    dataloaders/
        dataloader.py
        tabular.py  #tSNE plotting https://www.kaggle.com/philschmidt/quora-eda-model-selection-roc-pr-plots
        image.py #2D   - go.Scatter3d() tSNE plotting for dataset - https://www.kaggle.com/philschmidt/cervix-eda-model-selection#Image-neighbourhood
        image3D.py #3D image data

        class_overlap - multilabel classification
    models/
        Model.py
        tabularModel.py
        cnnModel.py
    predictions/
        analyze_preds.py   - ground truth known
            generateMetrics
            generateStatistics
        compareModel.py - takes list of preds_filenames, if None - provide list of predictions - ground truth unknown
            compareModelPred  - metrics, statistics
            compareModelPred_Stratified - stratified comparison
            scatterplot
            train_test_deviation

            generate_report
        ensemble.py
            average
            weighted
            find_weights
            NN_based
            Stacking

    metrics/
        classificationMetrics
        regressionMetrics
        segmentationMetrics

        classificationLoss
        regressionLoss
        segmentationLoss

    Utils/
        featureSelection.py
            Automated Feature Engineering Basics  - https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics?scriptVersionId=4120225
            https://www.featuretools.com/
        DownloadData.py
            fromKaggle
            fromTwitter
            fromGithub
            fromDrive
        getmodel_weights.py

        dataModifier.py
            Kfold_generator - https://www.kaggle.com/satishgunjal/tutorial-k-fold-cross-validation#Introduction-
            Startified_CV_generator - https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
        dicomHelper - dicomWindowding, DICOM_to_PNG , metadataExtractor

        explainability.py
            gradCam
            Shap
            Saliency maps

            generate_reports
        dimensionReduction.py
            tSNE - Image, tabular data
            PCA - Image, tabular data
            KNN clustering - Images, data
        medical_preproces.py
            segment_lungs
            image_registration
        augmentations.py - custom augmentations

        Images.py - nifti, tif --> png files


        CalculatePixelInfo  - rgb( mean and std) for your dataset






    constants.py - directories, model_weights_path

config/
credentials/
        weights_and_biases
        kaggle





type = "xgboost" , 'lightgbm'
problem = "classification" , "regression"
output = 'probabilities' , 'classes' , 'all_probabilities'




Add template functions -
    - custom Kfold
    - custom Stratified
    - custom ImageDataloader
    -

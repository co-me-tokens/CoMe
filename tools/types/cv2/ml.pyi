from __future__ import annotations
import cv2
__all__: list[str] = ['ANN_MLP', 'ANN_MLP_ANNEAL', 'ANN_MLP_BACKPROP', 'ANN_MLP_GAUSSIAN', 'ANN_MLP_IDENTITY', 'ANN_MLP_LEAKYRELU', 'ANN_MLP_NO_INPUT_SCALE', 'ANN_MLP_NO_OUTPUT_SCALE', 'ANN_MLP_RELU', 'ANN_MLP_RPROP', 'ANN_MLP_SIGMOID_SYM', 'ANN_MLP_UPDATE_WEIGHTS', 'ANN_MLP_create', 'ANN_MLP_load', 'BOOST_DISCRETE', 'BOOST_GENTLE', 'BOOST_LOGIT', 'BOOST_REAL', 'Boost', 'Boost_DISCRETE', 'Boost_GENTLE', 'Boost_LOGIT', 'Boost_REAL', 'Boost_create', 'Boost_load', 'COL_SAMPLE', 'DTREES_PREDICT_AUTO', 'DTREES_PREDICT_MASK', 'DTREES_PREDICT_MAX_VOTE', 'DTREES_PREDICT_SUM', 'DTrees', 'DTrees_PREDICT_AUTO', 'DTrees_PREDICT_MASK', 'DTrees_PREDICT_MAX_VOTE', 'DTrees_PREDICT_SUM', 'DTrees_create', 'DTrees_load', 'EM', 'EM_COV_MAT_DEFAULT', 'EM_COV_MAT_DIAGONAL', 'EM_COV_MAT_GENERIC', 'EM_COV_MAT_SPHERICAL', 'EM_DEFAULT_MAX_ITERS', 'EM_DEFAULT_NCLUSTERS', 'EM_START_AUTO_STEP', 'EM_START_E_STEP', 'EM_START_M_STEP', 'EM_create', 'EM_load', 'KNEAREST_BRUTE_FORCE', 'KNEAREST_KDTREE', 'KNearest', 'KNearest_BRUTE_FORCE', 'KNearest_KDTREE', 'KNearest_create', 'KNearest_load', 'LOGISTIC_REGRESSION_BATCH', 'LOGISTIC_REGRESSION_MINI_BATCH', 'LOGISTIC_REGRESSION_REG_DISABLE', 'LOGISTIC_REGRESSION_REG_L1', 'LOGISTIC_REGRESSION_REG_L2', 'LogisticRegression', 'LogisticRegression_BATCH', 'LogisticRegression_MINI_BATCH', 'LogisticRegression_REG_DISABLE', 'LogisticRegression_REG_L1', 'LogisticRegression_REG_L2', 'LogisticRegression_create', 'LogisticRegression_load', 'NormalBayesClassifier', 'NormalBayesClassifier_create', 'NormalBayesClassifier_load', 'ParamGrid', 'ParamGrid_create', 'ROW_SAMPLE', 'RTrees', 'RTrees_create', 'RTrees_load', 'STAT_MODEL_COMPRESSED_INPUT', 'STAT_MODEL_PREPROCESSED_INPUT', 'STAT_MODEL_RAW_OUTPUT', 'STAT_MODEL_UPDATE_MODEL', 'SVM', 'SVMSGD', 'SVMSGD_ASGD', 'SVMSGD_HARD_MARGIN', 'SVMSGD_SGD', 'SVMSGD_SOFT_MARGIN', 'SVMSGD_create', 'SVMSGD_load', 'SVM_C', 'SVM_CHI2', 'SVM_COEF', 'SVM_CUSTOM', 'SVM_C_SVC', 'SVM_DEGREE', 'SVM_EPS_SVR', 'SVM_GAMMA', 'SVM_INTER', 'SVM_LINEAR', 'SVM_NU', 'SVM_NU_SVC', 'SVM_NU_SVR', 'SVM_ONE_CLASS', 'SVM_P', 'SVM_POLY', 'SVM_RBF', 'SVM_SIGMOID', 'SVM_create', 'SVM_getDefaultGridPtr', 'SVM_load', 'StatModel', 'StatModel_COMPRESSED_INPUT', 'StatModel_PREPROCESSED_INPUT', 'StatModel_RAW_OUTPUT', 'StatModel_UPDATE_MODEL', 'TEST_ERROR', 'TRAIN_ERROR', 'TrainData', 'TrainData_create', 'TrainData_getSubMatrix', 'TrainData_getSubVector', 'VAR_CATEGORICAL', 'VAR_NUMERICAL', 'VAR_ORDERED']
class ANN_MLP(StatModel):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create() -> retval:
        """
        .   @brief Creates empty model
        .   
        .       Use StatModel::train to train the model, Algorithm::load\\<ANN_MLP\\>(filename) to load the pre-trained model.
        .       Note that the train method has optional flags: ANN_MLP::TrainFlags.
        """
    @staticmethod
    def getAnnealCoolingRatio() -> retval:
        """
        .   @see setAnnealCoolingRatio
        """
    @staticmethod
    def getAnnealFinalT() -> retval:
        """
        .   @see setAnnealFinalT
        """
    @staticmethod
    def getAnnealInitialT() -> retval:
        """
        .   @see setAnnealInitialT
        """
    @staticmethod
    def getAnnealItePerStep() -> retval:
        """
        .   @see setAnnealItePerStep
        """
    @staticmethod
    def getBackpropMomentumScale() -> retval:
        """
        .   @see setBackpropMomentumScale
        """
    @staticmethod
    def getBackpropWeightScale() -> retval:
        """
        .   @see setBackpropWeightScale
        """
    @staticmethod
    def getLayerSizes() -> retval:
        """
        .   Integer vector specifying the number of neurons in each layer including the input and output layers.
        .       The very first element specifies the number of elements in the input layer.
        .       The last element - number of elements in the output layer.
        .   @sa setLayerSizes
        """
    @staticmethod
    def getRpropDW0() -> retval:
        """
        .   @see setRpropDW0
        """
    @staticmethod
    def getRpropDWMax() -> retval:
        """
        .   @see setRpropDWMax
        """
    @staticmethod
    def getRpropDWMin() -> retval:
        """
        .   @see setRpropDWMin
        """
    @staticmethod
    def getRpropDWMinus() -> retval:
        """
        .   @see setRpropDWMinus
        """
    @staticmethod
    def getRpropDWPlus() -> retval:
        """
        .   @see setRpropDWPlus
        """
    @staticmethod
    def getTermCriteria() -> retval:
        """
        .   @see setTermCriteria
        """
    @staticmethod
    def getTrainMethod() -> retval:
        """
        .   Returns current training method
        """
    @staticmethod
    def getWeights(layerIdx) -> retval:
        """
        .
        """
    @staticmethod
    def load(filepath) -> retval:
        """
        .   @brief Loads and creates a serialized ANN from a file
        .        *
        .        * Use ANN::save to serialize and store an ANN to disk.
        .        * Load the ANN from this file again, by calling this function with the path to the file.
        .        *
        .        * @param filepath path to serialized ANN
        """
    @staticmethod
    def setActivationFunction(*args, **kwargs) -> None:
        """
        .   Initialize the activation function for each neuron.
        .       Currently the default and the only fully supported activation function is ANN_MLP::SIGMOID_SYM.
        .       @param type The type of activation function. See ANN_MLP::ActivationFunctions.
        .       @param param1 The first parameter of the activation function, \\f$\\alpha\\f$. Default value is 0.
        .       @param param2 The second parameter of the activation function, \\f$\\beta\\f$. Default value is 0.
        """
    @staticmethod
    def setAnnealCoolingRatio(val) -> None:
        """
        .   @copybrief getAnnealCoolingRatio @see getAnnealCoolingRatio
        """
    @staticmethod
    def setAnnealFinalT(val) -> None:
        """
        .   @copybrief getAnnealFinalT @see getAnnealFinalT
        """
    @staticmethod
    def setAnnealInitialT(val) -> None:
        """
        .   @copybrief getAnnealInitialT @see getAnnealInitialT
        """
    @staticmethod
    def setAnnealItePerStep(val) -> None:
        """
        .   @copybrief getAnnealItePerStep @see getAnnealItePerStep
        """
    @staticmethod
    def setBackpropMomentumScale(val) -> None:
        """
        .   @copybrief getBackpropMomentumScale @see getBackpropMomentumScale
        """
    @staticmethod
    def setBackpropWeightScale(val) -> None:
        """
        .   @copybrief getBackpropWeightScale @see getBackpropWeightScale
        """
    @staticmethod
    def setLayerSizes(_layer_sizes) -> None:
        """
        .   Integer vector specifying the number of neurons in each layer including the input and output layers.
        .       The very first element specifies the number of elements in the input layer.
        .       The last element - number of elements in the output layer. Default value is empty Mat.
        .   @sa getLayerSizes
        """
    @staticmethod
    def setRpropDW0(val) -> None:
        """
        .   @copybrief getRpropDW0 @see getRpropDW0
        """
    @staticmethod
    def setRpropDWMax(val) -> None:
        """
        .   @copybrief getRpropDWMax @see getRpropDWMax
        """
    @staticmethod
    def setRpropDWMin(val) -> None:
        """
        .   @copybrief getRpropDWMin @see getRpropDWMin
        """
    @staticmethod
    def setRpropDWMinus(val) -> None:
        """
        .   @copybrief getRpropDWMinus @see getRpropDWMinus
        """
    @staticmethod
    def setRpropDWPlus(val) -> None:
        """
        .   @copybrief getRpropDWPlus @see getRpropDWPlus
        """
    @staticmethod
    def setTermCriteria(val) -> None:
        """
        .   @copybrief getTermCriteria @see getTermCriteria
        """
    @staticmethod
    def setTrainMethod(*args, **kwargs) -> None:
        """
        .   Sets training method and common parameters.
        .       @param method Default value is ANN_MLP::RPROP. See ANN_MLP::TrainingMethods.
        .       @param param1 passed to setRpropDW0 for ANN_MLP::RPROP and to setBackpropWeightScale for ANN_MLP::BACKPROP and to initialT for ANN_MLP::ANNEAL.
        .       @param param2 passed to setRpropDWMin for ANN_MLP::RPROP and to setBackpropMomentumScale for ANN_MLP::BACKPROP and to finalT for ANN_MLP::ANNEAL.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Boost(DTrees):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create() -> retval:
        """
        .   Creates the empty model.
        .   Use StatModel::train to train the model, Algorithm::load\\<Boost\\>(filename) to load the pre-trained model.
        """
    @staticmethod
    def getBoostType() -> retval:
        """
        .   @see setBoostType
        """
    @staticmethod
    def getWeakCount() -> retval:
        """
        .   @see setWeakCount
        """
    @staticmethod
    def getWeightTrimRate() -> retval:
        """
        .   @see setWeightTrimRate
        """
    @staticmethod
    def load(*args, **kwargs) -> retval:
        """
        .   @brief Loads and creates a serialized Boost from a file
        .        *
        .        * Use Boost::save to serialize and store an RTree to disk.
        .        * Load the Boost from this file again, by calling this function with the path to the file.
        .        * Optionally specify the node for the file containing the classifier
        .        *
        .        * @param filepath path to serialized Boost
        .        * @param nodeName name of node containing the classifier
        """
    @staticmethod
    def setBoostType(val) -> None:
        """
        .   @copybrief getBoostType @see getBoostType
        """
    @staticmethod
    def setWeakCount(val) -> None:
        """
        .   @copybrief getWeakCount @see getWeakCount
        """
    @staticmethod
    def setWeightTrimRate(val) -> None:
        """
        .   @copybrief getWeightTrimRate @see getWeightTrimRate
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class DTrees(StatModel):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create() -> retval:
        """
        .   @brief Creates the empty model
        .   
        .       The static method creates empty decision tree with the specified parameters. It should be then
        .       trained using train method (see StatModel::train). Alternatively, you can load the model from
        .       file using Algorithm::load\\<DTrees\\>(filename).
        """
    @staticmethod
    def getCVFolds() -> retval:
        """
        .   @see setCVFolds
        """
    @staticmethod
    def getMaxCategories() -> retval:
        """
        .   @see setMaxCategories
        """
    @staticmethod
    def getMaxDepth() -> retval:
        """
        .   @see setMaxDepth
        """
    @staticmethod
    def getMinSampleCount() -> retval:
        """
        .   @see setMinSampleCount
        """
    @staticmethod
    def getPriors() -> retval:
        """
        .   @see setPriors
        """
    @staticmethod
    def getRegressionAccuracy() -> retval:
        """
        .   @see setRegressionAccuracy
        """
    @staticmethod
    def getTruncatePrunedTree() -> retval:
        """
        .   @see setTruncatePrunedTree
        """
    @staticmethod
    def getUse1SERule() -> retval:
        """
        .   @see setUse1SERule
        """
    @staticmethod
    def getUseSurrogates() -> retval:
        """
        .   @see setUseSurrogates
        """
    @staticmethod
    def load(*args, **kwargs) -> retval:
        """
        .   @brief Loads and creates a serialized DTrees from a file
        .        *
        .        * Use DTree::save to serialize and store an DTree to disk.
        .        * Load the DTree from this file again, by calling this function with the path to the file.
        .        * Optionally specify the node for the file containing the classifier
        .        *
        .        * @param filepath path to serialized DTree
        .        * @param nodeName name of node containing the classifier
        """
    @staticmethod
    def setCVFolds(val) -> None:
        """
        .   @copybrief getCVFolds @see getCVFolds
        """
    @staticmethod
    def setMaxCategories(val) -> None:
        """
        .   @copybrief getMaxCategories @see getMaxCategories
        """
    @staticmethod
    def setMaxDepth(val) -> None:
        """
        .   @copybrief getMaxDepth @see getMaxDepth
        """
    @staticmethod
    def setMinSampleCount(val) -> None:
        """
        .   @copybrief getMinSampleCount @see getMinSampleCount
        """
    @staticmethod
    def setPriors(val) -> None:
        """
        .   @copybrief getPriors @see getPriors
        """
    @staticmethod
    def setRegressionAccuracy(val) -> None:
        """
        .   @copybrief getRegressionAccuracy @see getRegressionAccuracy
        """
    @staticmethod
    def setTruncatePrunedTree(val) -> None:
        """
        .   @copybrief getTruncatePrunedTree @see getTruncatePrunedTree
        """
    @staticmethod
    def setUse1SERule(val) -> None:
        """
        .   @copybrief getUse1SERule @see getUse1SERule
        """
    @staticmethod
    def setUseSurrogates(val) -> None:
        """
        .   @copybrief getUseSurrogates @see getUseSurrogates
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class EM(StatModel):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create() -> retval:
        """
        .   Creates empty %EM model.
        .       The model should be trained then using StatModel::train(traindata, flags) method. Alternatively, you
        .       can use one of the EM::train\\* methods or load it from file using Algorithm::load\\<EM\\>(filename).
        """
    @staticmethod
    def getClustersNumber() -> retval:
        """
        .   @see setClustersNumber
        """
    @staticmethod
    def getCovarianceMatrixType() -> retval:
        """
        .   @see setCovarianceMatrixType
        """
    @staticmethod
    def getCovs(*args, **kwargs) -> covs:
        """
        .   @brief Returns covariation matrices
        .   
        .       Returns vector of covariation matrices. Number of matrices is the number of gaussian mixtures,
        .       each matrix is a square floating-point matrix NxN, where N is the space dimensionality.
        """
    @staticmethod
    def getMeans() -> retval:
        """
        .   @brief Returns the cluster centers (means of the Gaussian mixture)
        .   
        .       Returns matrix with the number of rows equal to the number of mixtures and number of columns
        .       equal to the space dimensionality.
        """
    @staticmethod
    def getTermCriteria() -> retval:
        """
        .   @see setTermCriteria
        """
    @staticmethod
    def getWeights() -> retval:
        """
        .   @brief Returns weights of the mixtures
        .   
        .       Returns vector with the number of elements equal to the number of mixtures.
        """
    @staticmethod
    def load(*args, **kwargs) -> retval:
        """
        .   @brief Loads and creates a serialized EM from a file
        .        *
        .        * Use EM::save to serialize and store an EM to disk.
        .        * Load the EM from this file again, by calling this function with the path to the file.
        .        * Optionally specify the node for the file containing the classifier
        .        *
        .        * @param filepath path to serialized EM
        .        * @param nodeName name of node containing the classifier
        """
    @staticmethod
    def predict(*args, **kwargs) -> retval, results:
        """
        .   @brief Returns posterior probabilities for the provided samples
        .   
        .       @param samples The input samples, floating-point matrix
        .       @param results The optional output \\f$ nSamples \\times nClusters\\f$ matrix of results. It contains
        .       posterior probabilities for each sample from the input
        .       @param flags This parameter will be ignored
        """
    @staticmethod
    def predict2(*args, **kwargs) -> retval, probs:
        """
        .   @brief Returns a likelihood logarithm value and an index of the most probable mixture component
        .       for the given sample.
        .   
        .       @param sample A sample for classification. It should be a one-channel matrix of
        .           \\f$1 \\times dims\\f$ or \\f$dims \\times 1\\f$ size.
        .       @param probs Optional output matrix that contains posterior probabilities of each component
        .           given the sample. It has \\f$1 \\times nclusters\\f$ size and CV_64FC1 type.
        .   
        .       The method returns a two-element double vector. Zero element is a likelihood logarithm value for
        .       the sample. First element is an index of the most probable mixture component for the given
        .       sample.
        """
    @staticmethod
    def setClustersNumber(val) -> None:
        """
        .   @copybrief getClustersNumber @see getClustersNumber
        """
    @staticmethod
    def setCovarianceMatrixType(val) -> None:
        """
        .   @copybrief getCovarianceMatrixType @see getCovarianceMatrixType
        """
    @staticmethod
    def setTermCriteria(val) -> None:
        """
        .   @copybrief getTermCriteria @see getTermCriteria
        """
    @staticmethod
    def trainE(*args, **kwargs) -> retval, logLikelihoods, labels, probs:
        """
        .   @brief Estimate the Gaussian mixture parameters from a samples set.
        .   
        .       This variation starts with Expectation step. You need to provide initial means \\f$a_k\\f$ of
        .       mixture components. Optionally you can pass initial weights \\f$\\pi_k\\f$ and covariance matrices
        .       \\f$S_k\\f$ of mixture components.
        .   
        .       @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
        .           one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
        .           it will be converted to the inner matrix of such type for the further computing.
        .       @param means0 Initial means \\f$a_k\\f$ of mixture components. It is a one-channel matrix of
        .           \\f$nclusters \\times dims\\f$ size. If the matrix does not have CV_64F type it will be
        .           converted to the inner matrix of such type for the further computing.
        .       @param covs0 The vector of initial covariance matrices \\f$S_k\\f$ of mixture components. Each of
        .           covariance matrices is a one-channel matrix of \\f$dims \\times dims\\f$ size. If the matrices
        .           do not have CV_64F type they will be converted to the inner matrices of such type for the
        .           further computing.
        .       @param weights0 Initial weights \\f$\\pi_k\\f$ of mixture components. It should be a one-channel
        .           floating-point matrix with \\f$1 \\times nclusters\\f$ or \\f$nclusters \\times 1\\f$ size.
        .       @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
        .           each sample. It has \\f$nsamples \\times 1\\f$ size and CV_64FC1 type.
        .       @param labels The optional output "class label" for each sample:
        .           \\f$\\texttt{labels}_i=\\texttt{arg max}_k(p_{i,k}), i=1..N\\f$ (indices of the most probable
        .           mixture component for each sample). It has \\f$nsamples \\times 1\\f$ size and CV_32SC1 type.
        .       @param probs The optional output matrix that contains posterior probabilities of each Gaussian
        .           mixture component given the each sample. It has \\f$nsamples \\times nclusters\\f$ size and
        .           CV_64FC1 type.
        """
    @staticmethod
    def trainEM(*args, **kwargs) -> retval, logLikelihoods, labels, probs:
        """
        .   @brief Estimate the Gaussian mixture parameters from a samples set.
        .   
        .       This variation starts with Expectation step. Initial values of the model parameters will be
        .       estimated by the k-means algorithm.
        .   
        .       Unlike many of the ML models, %EM is an unsupervised learning algorithm and it does not take
        .       responses (class labels or function values) as input. Instead, it computes the *Maximum
        .       Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the
        .       parameters inside the structure: \\f$p_{i,k}\\f$ in probs, \\f$a_k\\f$ in means , \\f$S_k\\f$ in
        .       covs[k], \\f$\\pi_k\\f$ in weights , and optionally computes the output "class label" for each
        .       sample: \\f$\\texttt{labels}_i=\\texttt{arg max}_k(p_{i,k}), i=1..N\\f$ (indices of the most
        .       probable mixture component for each sample).
        .   
        .       The trained model can be used further for prediction, just like any other classifier. The
        .       trained model is similar to the NormalBayesClassifier.
        .   
        .       @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
        .           one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
        .           it will be converted to the inner matrix of such type for the further computing.
        .       @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
        .           each sample. It has \\f$nsamples \\times 1\\f$ size and CV_64FC1 type.
        .       @param labels The optional output "class label" for each sample:
        .           \\f$\\texttt{labels}_i=\\texttt{arg max}_k(p_{i,k}), i=1..N\\f$ (indices of the most probable
        .           mixture component for each sample). It has \\f$nsamples \\times 1\\f$ size and CV_32SC1 type.
        .       @param probs The optional output matrix that contains posterior probabilities of each Gaussian
        .           mixture component given the each sample. It has \\f$nsamples \\times nclusters\\f$ size and
        .           CV_64FC1 type.
        """
    @staticmethod
    def trainM(*args, **kwargs) -> retval, logLikelihoods, labels, probs:
        """
        .   @brief Estimate the Gaussian mixture parameters from a samples set.
        .   
        .       This variation starts with Maximization step. You need to provide initial probabilities
        .       \\f$p_{i,k}\\f$ to use this option.
        .   
        .       @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
        .           one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type
        .           it will be converted to the inner matrix of such type for the further computing.
        .       @param probs0 the probabilities
        .       @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
        .           each sample. It has \\f$nsamples \\times 1\\f$ size and CV_64FC1 type.
        .       @param labels The optional output "class label" for each sample:
        .           \\f$\\texttt{labels}_i=\\texttt{arg max}_k(p_{i,k}), i=1..N\\f$ (indices of the most probable
        .           mixture component for each sample). It has \\f$nsamples \\times 1\\f$ size and CV_32SC1 type.
        .       @param probs The optional output matrix that contains posterior probabilities of each Gaussian
        .           mixture component given the each sample. It has \\f$nsamples \\times nclusters\\f$ size and
        .           CV_64FC1 type.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class KNearest(StatModel):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create() -> retval:
        """
        .   @brief Creates the empty model
        .   
        .       The static method creates empty %KNearest classifier. It should be then trained using StatModel::train method.
        """
    @staticmethod
    def findNearest(*args, **kwargs) -> retval, results, neighborResponses, dist:
        """
        .   @brief Finds the neighbors and predicts responses for input vectors.
        .   
        .       @param samples Input samples stored by rows. It is a single-precision floating-point matrix of
        .           `<number_of_samples> * k` size.
        .       @param k Number of used nearest neighbors. Should be greater than 1.
        .       @param results Vector with results of prediction (regression or classification) for each input
        .           sample. It is a single-precision floating-point vector with `<number_of_samples>` elements.
        .       @param neighborResponses Optional output values for corresponding neighbors. It is a single-
        .           precision floating-point matrix of `<number_of_samples> * k` size.
        .       @param dist Optional output distances from the input vectors to the corresponding neighbors. It
        .           is a single-precision floating-point matrix of `<number_of_samples> * k` size.
        .   
        .       For each input vector (a row of the matrix samples), the method finds the k nearest neighbors.
        .       In case of regression, the predicted result is a mean value of the particular vector's neighbor
        .       responses. In case of classification, the class is determined by voting.
        .   
        .       For each input vector, the neighbors are sorted by their distances to the vector.
        .   
        .       In case of C++ interface you can use output pointers to empty matrices and the function will
        .       allocate memory itself.
        .   
        .       If only a single input vector is passed, all output matrices are optional and the predicted
        .       value is returned by the method.
        .   
        .       The function is parallelized with the TBB library.
        """
    @staticmethod
    def getAlgorithmType() -> retval:
        """
        .   @see setAlgorithmType
        """
    @staticmethod
    def getDefaultK() -> retval:
        """
        .   @see setDefaultK
        """
    @staticmethod
    def getEmax() -> retval:
        """
        .   @see setEmax
        """
    @staticmethod
    def getIsClassifier() -> retval:
        """
        .   @see setIsClassifier
        """
    @staticmethod
    def load(filepath) -> retval:
        """
        .   @brief Loads and creates a serialized knearest from a file
        .        *
        .        * Use KNearest::save to serialize and store an KNearest to disk.
        .        * Load the KNearest from this file again, by calling this function with the path to the file.
        .        *
        .        * @param filepath path to serialized KNearest
        """
    @staticmethod
    def setAlgorithmType(val) -> None:
        """
        .   @copybrief getAlgorithmType @see getAlgorithmType
        """
    @staticmethod
    def setDefaultK(val) -> None:
        """
        .   @copybrief getDefaultK @see getDefaultK
        """
    @staticmethod
    def setEmax(val) -> None:
        """
        .   @copybrief getEmax @see getEmax
        """
    @staticmethod
    def setIsClassifier(val) -> None:
        """
        .   @copybrief getIsClassifier @see getIsClassifier
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class LogisticRegression(StatModel):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create() -> retval:
        """
        .   @brief Creates empty model.
        .   
        .       Creates Logistic Regression model with parameters given.
        """
    @staticmethod
    def getIterations() -> retval:
        """
        .   @see setIterations
        """
    @staticmethod
    def getLearningRate() -> retval:
        """
        .   @see setLearningRate
        """
    @staticmethod
    def getMiniBatchSize() -> retval:
        """
        .   @see setMiniBatchSize
        """
    @staticmethod
    def getRegularization() -> retval:
        """
        .   @see setRegularization
        """
    @staticmethod
    def getTermCriteria() -> retval:
        """
        .   @see setTermCriteria
        """
    @staticmethod
    def getTrainMethod() -> retval:
        """
        .   @see setTrainMethod
        """
    @staticmethod
    def get_learnt_thetas() -> retval:
        """
        .   @brief This function returns the trained parameters arranged across rows.
        .   
        .       For a two class classification problem, it returns a row matrix. It returns learnt parameters of
        .       the Logistic Regression as a matrix of type CV_32F.
        """
    @staticmethod
    def load(*args, **kwargs) -> retval:
        """
        .   @brief Loads and creates a serialized LogisticRegression from a file
        .        *
        .        * Use LogisticRegression::save to serialize and store an LogisticRegression to disk.
        .        * Load the LogisticRegression from this file again, by calling this function with the path to the file.
        .        * Optionally specify the node for the file containing the classifier
        .        *
        .        * @param filepath path to serialized LogisticRegression
        .        * @param nodeName name of node containing the classifier
        """
    @staticmethod
    def predict(*args, **kwargs) -> retval, results:
        """
        .   @brief Predicts responses for input samples and returns a float type.
        .   
        .       @param samples The input data for the prediction algorithm. Matrix [m x n], where each row
        .           contains variables (features) of one object being classified. Should have data type CV_32F.
        .       @param results Predicted labels as a column matrix of type CV_32S.
        .       @param flags Not used.
        """
    @staticmethod
    def setIterations(val) -> None:
        """
        .   @copybrief getIterations @see getIterations
        """
    @staticmethod
    def setLearningRate(val) -> None:
        """
        .   @copybrief getLearningRate @see getLearningRate
        """
    @staticmethod
    def setMiniBatchSize(val) -> None:
        """
        .   @copybrief getMiniBatchSize @see getMiniBatchSize
        """
    @staticmethod
    def setRegularization(val) -> None:
        """
        .   @copybrief getRegularization @see getRegularization
        """
    @staticmethod
    def setTermCriteria(val) -> None:
        """
        .   @copybrief getTermCriteria @see getTermCriteria
        """
    @staticmethod
    def setTrainMethod(val) -> None:
        """
        .   @copybrief getTrainMethod @see getTrainMethod
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class NormalBayesClassifier(StatModel):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create() -> retval:
        """
        .   Creates empty model
        .   Use StatModel::train to train the model after creation.
        """
    @staticmethod
    def load(*args, **kwargs) -> retval:
        """
        .   @brief Loads and creates a serialized NormalBayesClassifier from a file
        .        *
        .        * Use NormalBayesClassifier::save to serialize and store an NormalBayesClassifier to disk.
        .        * Load the NormalBayesClassifier from this file again, by calling this function with the path to the file.
        .        * Optionally specify the node for the file containing the classifier
        .        *
        .        * @param filepath path to serialized NormalBayesClassifier
        .        * @param nodeName name of node containing the classifier
        """
    @staticmethod
    def predictProb(*args, **kwargs) -> retval, outputs, outputProbs:
        """
        .   @brief Predicts the response for sample(s).
        .   
        .       The method estimates the most probable classes for input vectors. Input vectors (one or more)
        .       are stored as rows of the matrix inputs. In case of multiple input vectors, there should be one
        .       output vector outputs. The predicted class for a single input vector is returned by the method.
        .       The vector outputProbs contains the output probabilities corresponding to each element of
        .       result.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class ParamGrid:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create(*args, **kwargs) -> retval:
        """
        .   @brief Creates a ParamGrid Ptr that can be given to the %SVM::trainAuto method
        .   
        .       @param minVal minimum value of the parameter grid
        .       @param maxVal maximum value of the parameter grid
        .       @param logstep Logarithmic step for iterating the statmodel parameter
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class RTrees(DTrees):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create() -> retval:
        """
        .   Creates the empty model.
        .       Use StatModel::train to train the model, StatModel::train to create and train the model,
        .       Algorithm::load to load the pre-trained model.
        """
    @staticmethod
    def getActiveVarCount() -> retval:
        """
        .   @see setActiveVarCount
        """
    @staticmethod
    def getCalculateVarImportance() -> retval:
        """
        .   @see setCalculateVarImportance
        """
    @staticmethod
    def getOOBError() -> retval:
        """
        .   Returns the OOB error value, computed at the training stage when calcOOBError is set to true.
        .        * If this flag was set to false, 0 is returned. The OOB error is also scaled by sample weighting.
        """
    @staticmethod
    def getTermCriteria() -> retval:
        """
        .   @see setTermCriteria
        """
    @staticmethod
    def getVarImportance() -> retval:
        """
        .   Returns the variable importance array.
        .       The method returns the variable importance vector, computed at the training stage when
        .       CalculateVarImportance is set to true. If this flag was set to false, the empty matrix is
        .       returned.
        """
    @staticmethod
    def getVotes(*args, **kwargs) -> results:
        """
        .   Returns the result of each individual tree in the forest.
        .       In case the model is a regression problem, the method will return each of the trees'
        .       results for each of the sample cases. If the model is a classifier, it will return
        .       a Mat with samples + 1 rows, where the first row gives the class number and the
        .       following rows return the votes each class had for each sample.
        .           @param samples Array containing the samples for which votes will be calculated.
        .           @param results Array where the result of the calculation will be written.
        .           @param flags Flags for defining the type of RTrees.
        """
    @staticmethod
    def load(*args, **kwargs) -> retval:
        """
        .   @brief Loads and creates a serialized RTree from a file
        .        *
        .        * Use RTree::save to serialize and store an RTree to disk.
        .        * Load the RTree from this file again, by calling this function with the path to the file.
        .        * Optionally specify the node for the file containing the classifier
        .        *
        .        * @param filepath path to serialized RTree
        .        * @param nodeName name of node containing the classifier
        """
    @staticmethod
    def setActiveVarCount(val) -> None:
        """
        .   @copybrief getActiveVarCount @see getActiveVarCount
        """
    @staticmethod
    def setCalculateVarImportance(val) -> None:
        """
        .   @copybrief getCalculateVarImportance @see getCalculateVarImportance
        """
    @staticmethod
    def setTermCriteria(val) -> None:
        """
        .   @copybrief getTermCriteria @see getTermCriteria
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class SVM(StatModel):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create() -> retval:
        """
        .   Creates empty model.
        .       Use StatModel::train to train the model. Since %SVM has several parameters, you may want to
        .   find the best parameters for your problem, it can be done with SVM::trainAuto.
        """
    @staticmethod
    def getC() -> retval:
        """
        .   @see setC
        """
    @staticmethod
    def getClassWeights() -> retval:
        """
        .   @see setClassWeights
        """
    @staticmethod
    def getCoef0() -> retval:
        """
        .   @see setCoef0
        """
    @staticmethod
    def getDecisionFunction(*args, **kwargs) -> retval, alpha, svidx:
        """
        .   @brief Retrieves the decision function
        .   
        .       @param i the index of the decision function. If the problem solved is regression, 1-class or
        .           2-class classification, then there will be just one decision function and the index should
        .           always be 0. Otherwise, in the case of N-class classification, there will be \\f$N(N-1)/2\\f$
        .           decision functions.
        .       @param alpha the optional output vector for weights, corresponding to different support vectors.
        .           In the case of linear %SVM all the alpha's will be 1's.
        .       @param svidx the optional output vector of indices of support vectors within the matrix of
        .           support vectors (which can be retrieved by SVM::getSupportVectors). In the case of linear
        .           %SVM each decision function consists of a single "compressed" support vector.
        .   
        .       The method returns rho parameter of the decision function, a scalar subtracted from the weighted
        .       sum of kernel responses.
        """
    @staticmethod
    def getDefaultGridPtr(param_id) -> retval:
        """
        .   @brief Generates a grid for %SVM parameters.
        .   
        .       @param param_id %SVM parameters IDs that must be one of the SVM::ParamTypes. The grid is
        .       generated for the parameter with this ID.
        .   
        .       The function generates a grid pointer for the specified parameter of the %SVM algorithm.
        .       The grid may be passed to the function SVM::trainAuto.
        """
    @staticmethod
    def getDegree() -> retval:
        """
        .   @see setDegree
        """
    @staticmethod
    def getGamma() -> retval:
        """
        .   @see setGamma
        """
    @staticmethod
    def getKernelType() -> retval:
        """
        .   Type of a %SVM kernel.
        .   See SVM::KernelTypes. Default value is SVM::RBF.
        """
    @staticmethod
    def getNu() -> retval:
        """
        .   @see setNu
        """
    @staticmethod
    def getP() -> retval:
        """
        .   @see setP
        """
    @staticmethod
    def getSupportVectors() -> retval:
        """
        .   @brief Retrieves all the support vectors
        .   
        .       The method returns all the support vectors as a floating-point matrix, where support vectors are
        .       stored as matrix rows.
        """
    @staticmethod
    def getTermCriteria() -> retval:
        """
        .   @see setTermCriteria
        """
    @staticmethod
    def getType() -> retval:
        """
        .   @see setType
        """
    @staticmethod
    def getUncompressedSupportVectors() -> retval:
        """
        .   @brief Retrieves all the uncompressed support vectors of a linear %SVM
        .   
        .       The method returns all the uncompressed support vectors of a linear %SVM that the compressed
        .       support vector, used for prediction, was derived from. They are returned in a floating-point
        .       matrix, where the support vectors are stored as matrix rows.
        """
    @staticmethod
    def load(filepath) -> retval:
        """
        .   @brief Loads and creates a serialized svm from a file
        .        *
        .        * Use SVM::save to serialize and store an SVM to disk.
        .        * Load the SVM from this file again, by calling this function with the path to the file.
        .        *
        .        * @param filepath path to serialized svm
        """
    @staticmethod
    def setC(val) -> None:
        """
        .   @copybrief getC @see getC
        """
    @staticmethod
    def setClassWeights(val) -> None:
        """
        .   @copybrief getClassWeights @see getClassWeights
        """
    @staticmethod
    def setCoef0(val) -> None:
        """
        .   @copybrief getCoef0 @see getCoef0
        """
    @staticmethod
    def setDegree(val) -> None:
        """
        .   @copybrief getDegree @see getDegree
        """
    @staticmethod
    def setGamma(val) -> None:
        """
        .   @copybrief getGamma @see getGamma
        """
    @staticmethod
    def setKernel(kernelType) -> None:
        """
        .   Initialize with one of predefined kernels.
        .   See SVM::KernelTypes.
        """
    @staticmethod
    def setNu(val) -> None:
        """
        .   @copybrief getNu @see getNu
        """
    @staticmethod
    def setP(val) -> None:
        """
        .   @copybrief getP @see getP
        """
    @staticmethod
    def setTermCriteria(val) -> None:
        """
        .   @copybrief getTermCriteria @see getTermCriteria
        """
    @staticmethod
    def setType(val) -> None:
        """
        .   @copybrief getType @see getType
        """
    @staticmethod
    def trainAuto(*args, **kwargs) -> retval:
        """
        .   @brief Trains an %SVM with optimal parameters
        .   
        .       @param samples training samples
        .       @param layout See ml::SampleTypes.
        .       @param responses vector of responses associated with the training samples.
        .       @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
        .           subset is used to test the model, the others form the train set. So, the %SVM algorithm is
        .       @param Cgrid grid for C
        .       @param gammaGrid grid for gamma
        .       @param pGrid grid for p
        .       @param nuGrid grid for nu
        .       @param coeffGrid grid for coeff
        .       @param degreeGrid grid for degree
        .       @param balanced If true and the problem is 2-class classification then the method creates more
        .           balanced cross-validation subsets that is proportions between classes in subsets are close
        .           to such proportion in the whole train dataset.
        .   
        .       The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p,
        .       nu, coef0, degree. Parameters are considered optimal when the cross-validation
        .       estimate of the test set error is minimal.
        .   
        .       This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only
        .       offers rudimentary parameter options.
        .   
        .       This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the
        .       regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and
        .       the usual %SVM with parameters specified in params is executed.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class SVMSGD(StatModel):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create() -> retval:
        """
        .   @brief Creates empty model.
        .        * Use StatModel::train to train the model. Since %SVMSGD has several parameters, you may want to
        .        * find the best parameters for your problem or use setOptimalParameters() to set some default parameters.
        """
    @staticmethod
    def getInitialStepSize() -> retval:
        """
        .   @see setInitialStepSize
        """
    @staticmethod
    def getMarginRegularization() -> retval:
        """
        .   @see setMarginRegularization
        """
    @staticmethod
    def getMarginType() -> retval:
        """
        .   @see setMarginType
        """
    @staticmethod
    def getShift() -> retval:
        """
        .   * @return the shift of the trained model (decision function f(x) = weights * x + shift).
        """
    @staticmethod
    def getStepDecreasingPower() -> retval:
        """
        .   @see setStepDecreasingPower
        """
    @staticmethod
    def getSvmsgdType() -> retval:
        """
        .   @see setSvmsgdType
        """
    @staticmethod
    def getTermCriteria() -> retval:
        """
        .   @see setTermCriteria
        """
    @staticmethod
    def getWeights() -> retval:
        """
        .   * @return the weights of the trained model (decision function f(x) = weights * x + shift).
        """
    @staticmethod
    def load(*args, **kwargs) -> retval:
        """
        .   @brief Loads and creates a serialized SVMSGD from a file
        .        *
        .        * Use SVMSGD::save to serialize and store an SVMSGD to disk.
        .        * Load the SVMSGD from this file again, by calling this function with the path to the file.
        .        * Optionally specify the node for the file containing the classifier
        .        *
        .        * @param filepath path to serialized SVMSGD
        .        * @param nodeName name of node containing the classifier
        """
    @staticmethod
    def setInitialStepSize(InitialStepSize) -> None:
        """
        .   @copybrief getInitialStepSize @see getInitialStepSize
        """
    @staticmethod
    def setMarginRegularization(marginRegularization) -> None:
        """
        .   @copybrief getMarginRegularization @see getMarginRegularization
        """
    @staticmethod
    def setMarginType(marginType) -> None:
        """
        .   @copybrief getMarginType @see getMarginType
        """
    @staticmethod
    def setOptimalParameters(*args, **kwargs) -> None:
        """
        .   @brief Function sets optimal parameters values for chosen SVM SGD model.
        .        * @param svmsgdType is the type of SVMSGD classifier.
        .        * @param marginType is the type of margin constraint.
        """
    @staticmethod
    def setStepDecreasingPower(stepDecreasingPower) -> None:
        """
        .   @copybrief getStepDecreasingPower @see getStepDecreasingPower
        """
    @staticmethod
    def setSvmsgdType(svmsgdType) -> None:
        """
        .   @copybrief getSvmsgdType @see getSvmsgdType
        """
    @staticmethod
    def setTermCriteria(val) -> None:
        """
        .   @copybrief getTermCriteria @see getTermCriteria
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class StatModel(cv2.Algorithm):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def calcError(*args, **kwargs) -> retval, resp:
        """
        .   @brief Computes error on the training or test dataset
        .   
        .       @param data the training data
        .       @param test if true, the error is computed over the test subset of the data, otherwise it's
        .           computed over the training subset of the data. Please note that if you loaded a completely
        .           different dataset to evaluate already trained classifier, you will probably want not to set
        .           the test subset at all with TrainData::setTrainTestSplitRatio and specify test=false, so
        .           that the error is computed for the whole new set. Yes, this sounds a bit confusing.
        .       @param resp the optional output responses.
        .   
        .       The method uses StatModel::predict to compute the error. For regression models the error is
        .       computed as RMS, for classifiers - as a percent of missclassified samples (0%-100%).
        """
    @staticmethod
    def empty() -> retval:
        """
        .
        """
    @staticmethod
    def getVarCount() -> retval:
        """
        .   @brief Returns the number of variables in training samples
        """
    @staticmethod
    def isClassifier() -> retval:
        """
        .   @brief Returns true if the model is classifier
        """
    @staticmethod
    def isTrained() -> retval:
        """
        .   @brief Returns true if the model is trained
        """
    @staticmethod
    def predict(*args, **kwargs) -> retval, results:
        """
        .   @brief Predicts response(s) for the provided sample(s)
        .   
        .       @param samples The input samples, floating-point matrix
        .       @param results The optional output matrix of results.
        .       @param flags The optional flags, model-dependent. See cv::ml::StatModel::Flags.
        """
    @staticmethod
    def train(*args, **kwargs) -> retval:
        """
        .   @brief Trains the statistical model
        .   
        .       @param trainData training data that can be loaded from file using TrainData::loadFromCSV or
        .           created with TrainData::create.
        .       @param flags optional flags, depending on the model. Some of the models can be updated with the
        .           new training samples, not completely overwritten (such as NormalBayesClassifier or ANN_MLP).
        
        
        
        train(samples, layout, responses) -> retval
        .   @brief Trains the statistical model
        .   
        .       @param samples training samples
        .       @param layout See ml::SampleTypes.
        .       @param responses vector of responses associated with the training samples.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class TrainData:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def create(*args, **kwargs) -> retval:
        """
        .   @brief Creates training data from in-memory arrays.
        .   
        .       @param samples matrix of samples. It should have CV_32F type.
        .       @param layout see ml::SampleTypes.
        .       @param responses matrix of responses. If the responses are scalar, they should be stored as a
        .           single row or as a single column. The matrix should have type CV_32F or CV_32S (in the
        .           former case the responses are considered as ordered by default; in the latter case - as
        .           categorical)
        .       @param varIdx vector specifying which variables to use for training. It can be an integer vector
        .           (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of
        .           active variables.
        .       @param sampleIdx vector specifying which samples to use for training. It can be an integer
        .           vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask
        .           of training samples.
        .       @param sampleWeights optional vector with weights for each sample. It should have CV_32F type.
        .       @param varType optional vector of type CV_8U and size `<number_of_variables_in_samples> +
        .           <number_of_variables_in_responses>`, containing types of each input and output variable. See
        .           ml::VariableTypes.
        """
    @staticmethod
    def getCatCount(vi) -> retval:
        """
        .
        """
    @staticmethod
    def getCatMap() -> retval:
        """
        .
        """
    @staticmethod
    def getCatOfs() -> retval:
        """
        .
        """
    @staticmethod
    def getClassLabels() -> retval:
        """
        .   @brief Returns the vector of class labels
        .   
        .       The function returns vector of unique labels occurred in the responses.
        """
    @staticmethod
    def getDefaultSubstValues() -> retval:
        """
        .
        """
    @staticmethod
    def getLayout() -> retval:
        """
        .
        """
    @staticmethod
    def getMissing() -> retval:
        """
        .
        """
    @staticmethod
    def getNAllVars() -> retval:
        """
        .
        """
    @staticmethod
    def getNSamples() -> retval:
        """
        .
        """
    @staticmethod
    def getNTestSamples() -> retval:
        """
        .
        """
    @staticmethod
    def getNTrainSamples() -> retval:
        """
        .
        """
    @staticmethod
    def getNVars() -> retval:
        """
        .
        """
    @staticmethod
    def getNames(names) -> None:
        """
        .   @brief Returns vector of symbolic names captured in loadFromCSV()
        """
    @staticmethod
    def getNormCatResponses() -> retval:
        """
        .
        """
    @staticmethod
    def getResponseType() -> retval:
        """
        .
        """
    @staticmethod
    def getResponses() -> retval:
        """
        .
        """
    @staticmethod
    def getSample(varIdx, sidx, buf) -> None:
        """
        .
        """
    @staticmethod
    def getSampleWeights() -> retval:
        """
        .
        """
    @staticmethod
    def getSamples() -> retval:
        """
        .
        """
    @staticmethod
    def getSubMatrix(matrix, idx, layout) -> retval:
        """
        .   @brief Extract from matrix rows/cols specified by passed indexes.
        .       @param matrix input matrix (supported types: CV_32S, CV_32F, CV_64F)
        .       @param idx 1D index vector
        .       @param layout specifies to extract rows (cv::ml::ROW_SAMPLES) or to extract columns (cv::ml::COL_SAMPLES)
        """
    @staticmethod
    def getSubVector(vec, idx) -> retval:
        """
        .   @brief Extract from 1D vector elements specified by passed indexes.
        .       @param vec input vector (supported types: CV_32S, CV_32F, CV_64F)
        .       @param idx 1D index vector
        """
    @staticmethod
    def getTestNormCatResponses() -> retval:
        """
        .
        """
    @staticmethod
    def getTestResponses() -> retval:
        """
        .
        """
    @staticmethod
    def getTestSampleIdx() -> retval:
        """
        .
        """
    @staticmethod
    def getTestSampleWeights() -> retval:
        """
        .
        """
    @staticmethod
    def getTestSamples() -> retval:
        """
        .   @brief Returns matrix of test samples
        """
    @staticmethod
    def getTrainNormCatResponses() -> retval:
        """
        .   @brief Returns the vector of normalized categorical responses
        .   
        .       The function returns vector of responses. Each response is integer from `0` to `<number of
        .       classes>-1`. The actual label value can be retrieved then from the class label vector, see
        .       TrainData::getClassLabels.
        """
    @staticmethod
    def getTrainResponses() -> retval:
        """
        .   @brief Returns the vector of responses
        .   
        .       The function returns ordered or the original categorical responses. Usually it's used in
        .       regression algorithms.
        """
    @staticmethod
    def getTrainSampleIdx() -> retval:
        """
        .
        """
    @staticmethod
    def getTrainSampleWeights() -> retval:
        """
        .
        """
    @staticmethod
    def getTrainSamples(*args, **kwargs) -> retval:
        """
        .   @brief Returns matrix of train samples
        .   
        .       @param layout The requested layout. If it's different from the initial one, the matrix is
        .           transposed. See ml::SampleTypes.
        .       @param compressSamples if true, the function returns only the training samples (specified by
        .           sampleIdx)
        .       @param compressVars if true, the function returns the shorter training samples, containing only
        .           the active variables.
        .   
        .       In current implementation the function tries to avoid physical data copying and returns the
        .       matrix stored inside TrainData (unless the transposition or compression is needed).
        """
    @staticmethod
    def getValues(vi, sidx, values) -> None:
        """
        .
        """
    @staticmethod
    def getVarIdx() -> retval:
        """
        .
        """
    @staticmethod
    def getVarSymbolFlags() -> retval:
        """
        .
        """
    @staticmethod
    def getVarType() -> retval:
        """
        .
        """
    @staticmethod
    def setTrainTestSplit(*args, **kwargs) -> None:
        """
        .   @brief Splits the training data into the training and test parts
        .       @sa TrainData::setTrainTestSplitRatio
        """
    @staticmethod
    def setTrainTestSplitRatio(*args, **kwargs) -> None:
        """
        .   @brief Splits the training data into the training and test parts
        .   
        .       The function selects a subset of specified relative size and then returns it as the training
        .       set. If the function is not called, all the data is used for training. Please, note that for
        .       each of TrainData::getTrain\\* there is corresponding TrainData::getTest\\*, so that the test
        .       subset can be retrieved and processed as well.
        .       @sa TrainData::setTrainTestSplit
        """
    @staticmethod
    def shuffleTrainTest() -> None:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
def ANN_MLP_create() -> retval:
    """
    .   @brief Creates empty model
    .   
    .       Use StatModel::train to train the model, Algorithm::load\\<ANN_MLP\\>(filename) to load the pre-trained model.
    .       Note that the train method has optional flags: ANN_MLP::TrainFlags.
    """
def ANN_MLP_load(filepath) -> retval:
    """
    .   @brief Loads and creates a serialized ANN from a file
    .        *
    .        * Use ANN::save to serialize and store an ANN to disk.
    .        * Load the ANN from this file again, by calling this function with the path to the file.
    .        *
    .        * @param filepath path to serialized ANN
    """
def Boost_create() -> retval:
    """
    .   Creates the empty model.
    .   Use StatModel::train to train the model, Algorithm::load\\<Boost\\>(filename) to load the pre-trained model.
    """
def Boost_load(*args, **kwargs) -> retval:
    """
    .   @brief Loads and creates a serialized Boost from a file
    .        *
    .        * Use Boost::save to serialize and store an RTree to disk.
    .        * Load the Boost from this file again, by calling this function with the path to the file.
    .        * Optionally specify the node for the file containing the classifier
    .        *
    .        * @param filepath path to serialized Boost
    .        * @param nodeName name of node containing the classifier
    """
def DTrees_create() -> retval:
    """
    .   @brief Creates the empty model
    .   
    .       The static method creates empty decision tree with the specified parameters. It should be then
    .       trained using train method (see StatModel::train). Alternatively, you can load the model from
    .       file using Algorithm::load\\<DTrees\\>(filename).
    """
def DTrees_load(*args, **kwargs) -> retval:
    """
    .   @brief Loads and creates a serialized DTrees from a file
    .        *
    .        * Use DTree::save to serialize and store an DTree to disk.
    .        * Load the DTree from this file again, by calling this function with the path to the file.
    .        * Optionally specify the node for the file containing the classifier
    .        *
    .        * @param filepath path to serialized DTree
    .        * @param nodeName name of node containing the classifier
    """
def EM_create() -> retval:
    """
    .   Creates empty %EM model.
    .       The model should be trained then using StatModel::train(traindata, flags) method. Alternatively, you
    .       can use one of the EM::train\\* methods or load it from file using Algorithm::load\\<EM\\>(filename).
    """
def EM_load(*args, **kwargs) -> retval:
    """
    .   @brief Loads and creates a serialized EM from a file
    .        *
    .        * Use EM::save to serialize and store an EM to disk.
    .        * Load the EM from this file again, by calling this function with the path to the file.
    .        * Optionally specify the node for the file containing the classifier
    .        *
    .        * @param filepath path to serialized EM
    .        * @param nodeName name of node containing the classifier
    """
def KNearest_create() -> retval:
    """
    .   @brief Creates the empty model
    .   
    .       The static method creates empty %KNearest classifier. It should be then trained using StatModel::train method.
    """
def KNearest_load(filepath) -> retval:
    """
    .   @brief Loads and creates a serialized knearest from a file
    .        *
    .        * Use KNearest::save to serialize and store an KNearest to disk.
    .        * Load the KNearest from this file again, by calling this function with the path to the file.
    .        *
    .        * @param filepath path to serialized KNearest
    """
def LogisticRegression_create() -> retval:
    """
    .   @brief Creates empty model.
    .   
    .       Creates Logistic Regression model with parameters given.
    """
def LogisticRegression_load(*args, **kwargs) -> retval:
    """
    .   @brief Loads and creates a serialized LogisticRegression from a file
    .        *
    .        * Use LogisticRegression::save to serialize and store an LogisticRegression to disk.
    .        * Load the LogisticRegression from this file again, by calling this function with the path to the file.
    .        * Optionally specify the node for the file containing the classifier
    .        *
    .        * @param filepath path to serialized LogisticRegression
    .        * @param nodeName name of node containing the classifier
    """
def NormalBayesClassifier_create() -> retval:
    """
    .   Creates empty model
    .   Use StatModel::train to train the model after creation.
    """
def NormalBayesClassifier_load(*args, **kwargs) -> retval:
    """
    .   @brief Loads and creates a serialized NormalBayesClassifier from a file
    .        *
    .        * Use NormalBayesClassifier::save to serialize and store an NormalBayesClassifier to disk.
    .        * Load the NormalBayesClassifier from this file again, by calling this function with the path to the file.
    .        * Optionally specify the node for the file containing the classifier
    .        *
    .        * @param filepath path to serialized NormalBayesClassifier
    .        * @param nodeName name of node containing the classifier
    """
def ParamGrid_create(*args, **kwargs) -> retval:
    """
    .   @brief Creates a ParamGrid Ptr that can be given to the %SVM::trainAuto method
    .   
    .       @param minVal minimum value of the parameter grid
    .       @param maxVal maximum value of the parameter grid
    .       @param logstep Logarithmic step for iterating the statmodel parameter
    """
def RTrees_create() -> retval:
    """
    .   Creates the empty model.
    .       Use StatModel::train to train the model, StatModel::train to create and train the model,
    .       Algorithm::load to load the pre-trained model.
    """
def RTrees_load(*args, **kwargs) -> retval:
    """
    .   @brief Loads and creates a serialized RTree from a file
    .        *
    .        * Use RTree::save to serialize and store an RTree to disk.
    .        * Load the RTree from this file again, by calling this function with the path to the file.
    .        * Optionally specify the node for the file containing the classifier
    .        *
    .        * @param filepath path to serialized RTree
    .        * @param nodeName name of node containing the classifier
    """
def SVMSGD_create() -> retval:
    """
    .   @brief Creates empty model.
    .        * Use StatModel::train to train the model. Since %SVMSGD has several parameters, you may want to
    .        * find the best parameters for your problem or use setOptimalParameters() to set some default parameters.
    """
def SVMSGD_load(*args, **kwargs) -> retval:
    """
    .   @brief Loads and creates a serialized SVMSGD from a file
    .        *
    .        * Use SVMSGD::save to serialize and store an SVMSGD to disk.
    .        * Load the SVMSGD from this file again, by calling this function with the path to the file.
    .        * Optionally specify the node for the file containing the classifier
    .        *
    .        * @param filepath path to serialized SVMSGD
    .        * @param nodeName name of node containing the classifier
    """
def SVM_create() -> retval:
    """
    .   Creates empty model.
    .       Use StatModel::train to train the model. Since %SVM has several parameters, you may want to
    .   find the best parameters for your problem, it can be done with SVM::trainAuto.
    """
def SVM_getDefaultGridPtr(param_id) -> retval:
    """
    .   @brief Generates a grid for %SVM parameters.
    .   
    .       @param param_id %SVM parameters IDs that must be one of the SVM::ParamTypes. The grid is
    .       generated for the parameter with this ID.
    .   
    .       The function generates a grid pointer for the specified parameter of the %SVM algorithm.
    .       The grid may be passed to the function SVM::trainAuto.
    """
def SVM_load(filepath) -> retval:
    """
    .   @brief Loads and creates a serialized svm from a file
    .        *
    .        * Use SVM::save to serialize and store an SVM to disk.
    .        * Load the SVM from this file again, by calling this function with the path to the file.
    .        *
    .        * @param filepath path to serialized svm
    """
def TrainData_create(*args, **kwargs) -> retval:
    """
    .   @brief Creates training data from in-memory arrays.
    .   
    .       @param samples matrix of samples. It should have CV_32F type.
    .       @param layout see ml::SampleTypes.
    .       @param responses matrix of responses. If the responses are scalar, they should be stored as a
    .           single row or as a single column. The matrix should have type CV_32F or CV_32S (in the
    .           former case the responses are considered as ordered by default; in the latter case - as
    .           categorical)
    .       @param varIdx vector specifying which variables to use for training. It can be an integer vector
    .           (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of
    .           active variables.
    .       @param sampleIdx vector specifying which samples to use for training. It can be an integer
    .           vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask
    .           of training samples.
    .       @param sampleWeights optional vector with weights for each sample. It should have CV_32F type.
    .       @param varType optional vector of type CV_8U and size `<number_of_variables_in_samples> +
    .           <number_of_variables_in_responses>`, containing types of each input and output variable. See
    .           ml::VariableTypes.
    """
def TrainData_getSubMatrix(matrix, idx, layout) -> retval:
    """
    .   @brief Extract from matrix rows/cols specified by passed indexes.
    .       @param matrix input matrix (supported types: CV_32S, CV_32F, CV_64F)
    .       @param idx 1D index vector
    .       @param layout specifies to extract rows (cv::ml::ROW_SAMPLES) or to extract columns (cv::ml::COL_SAMPLES)
    """
def TrainData_getSubVector(vec, idx) -> retval:
    """
    .   @brief Extract from 1D vector elements specified by passed indexes.
    .       @param vec input vector (supported types: CV_32S, CV_32F, CV_64F)
    .       @param idx 1D index vector
    """
ANN_MLP_ANNEAL: int = 2
ANN_MLP_BACKPROP: int = 0
ANN_MLP_GAUSSIAN: int = 2
ANN_MLP_IDENTITY: int = 0
ANN_MLP_LEAKYRELU: int = 4
ANN_MLP_NO_INPUT_SCALE: int = 2
ANN_MLP_NO_OUTPUT_SCALE: int = 4
ANN_MLP_RELU: int = 3
ANN_MLP_RPROP: int = 1
ANN_MLP_SIGMOID_SYM: int = 1
ANN_MLP_UPDATE_WEIGHTS: int = 1
BOOST_DISCRETE: int = 0
BOOST_GENTLE: int = 3
BOOST_LOGIT: int = 2
BOOST_REAL: int = 1
Boost_DISCRETE: int = 0
Boost_GENTLE: int = 3
Boost_LOGIT: int = 2
Boost_REAL: int = 1
COL_SAMPLE: int = 1
DTREES_PREDICT_AUTO: int = 0
DTREES_PREDICT_MASK: int = 768
DTREES_PREDICT_MAX_VOTE: int = 512
DTREES_PREDICT_SUM: int = 256
DTrees_PREDICT_AUTO: int = 0
DTrees_PREDICT_MASK: int = 768
DTrees_PREDICT_MAX_VOTE: int = 512
DTrees_PREDICT_SUM: int = 256
EM_COV_MAT_DEFAULT: int = 1
EM_COV_MAT_DIAGONAL: int = 1
EM_COV_MAT_GENERIC: int = 2
EM_COV_MAT_SPHERICAL: int = 0
EM_DEFAULT_MAX_ITERS: int = 100
EM_DEFAULT_NCLUSTERS: int = 5
EM_START_AUTO_STEP: int = 0
EM_START_E_STEP: int = 1
EM_START_M_STEP: int = 2
KNEAREST_BRUTE_FORCE: int = 1
KNEAREST_KDTREE: int = 2
KNearest_BRUTE_FORCE: int = 1
KNearest_KDTREE: int = 2
LOGISTIC_REGRESSION_BATCH: int = 0
LOGISTIC_REGRESSION_MINI_BATCH: int = 1
LOGISTIC_REGRESSION_REG_DISABLE: int = -1
LOGISTIC_REGRESSION_REG_L1: int = 0
LOGISTIC_REGRESSION_REG_L2: int = 1
LogisticRegression_BATCH: int = 0
LogisticRegression_MINI_BATCH: int = 1
LogisticRegression_REG_DISABLE: int = -1
LogisticRegression_REG_L1: int = 0
LogisticRegression_REG_L2: int = 1
ROW_SAMPLE: int = 0
STAT_MODEL_COMPRESSED_INPUT: int = 2
STAT_MODEL_PREPROCESSED_INPUT: int = 4
STAT_MODEL_RAW_OUTPUT: int = 1
STAT_MODEL_UPDATE_MODEL: int = 1
SVMSGD_ASGD: int = 1
SVMSGD_HARD_MARGIN: int = 1
SVMSGD_SGD: int = 0
SVMSGD_SOFT_MARGIN: int = 0
SVM_C: int = 0
SVM_CHI2: int = 4
SVM_COEF: int = 4
SVM_CUSTOM: int = -1
SVM_C_SVC: int = 100
SVM_DEGREE: int = 5
SVM_EPS_SVR: int = 103
SVM_GAMMA: int = 1
SVM_INTER: int = 5
SVM_LINEAR: int = 0
SVM_NU: int = 3
SVM_NU_SVC: int = 101
SVM_NU_SVR: int = 104
SVM_ONE_CLASS: int = 102
SVM_P: int = 2
SVM_POLY: int = 1
SVM_RBF: int = 2
SVM_SIGMOID: int = 3
StatModel_COMPRESSED_INPUT: int = 2
StatModel_PREPROCESSED_INPUT: int = 4
StatModel_RAW_OUTPUT: int = 1
StatModel_UPDATE_MODEL: int = 1
TEST_ERROR: int = 0
TRAIN_ERROR: int = 1
VAR_CATEGORICAL: int = 1
VAR_NUMERICAL: int = 0
VAR_ORDERED: int = 0

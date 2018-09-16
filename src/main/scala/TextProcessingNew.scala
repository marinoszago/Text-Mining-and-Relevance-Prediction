import MlModels.{DecisionTree_Regressor,RandomForest_Regressor,Linear_Regressor}
import TextProcessingController.{DataframeCreatorController,SimilarityMethods}
import com.sun.javafx.binding.SelectBinding.AsInteger
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.{SparseVector, Vector, Vectors}

import scala.collection.mutable
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, MatrixEntry, RowMatrix}

import scala.collection.mutable.ListBuffer
//import org.apache.spark.mllib.linalg.VectorUDT
import org.apache.spark.ml.linalg.VectorUDT

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.feature.NGram



object TextProcessingNew {


  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setMaster("local[*]")
      .setAppName("Text Mining Project").setSparkHome("src/main/resources")
    conf.set("spark.hadoop.validateOutputSpecs", "false")
    val sc = new SparkContext(conf)
    val ss = SparkSession.builder().master("local[*]").appName("Text Mining Project").getOrCreate()
    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    val dataframeCreatorController = new DataframeCreatorController()

    /** testDF : result columns : rProductUID , rFilteredWords (all search terms filtered grouped by rProductUID) */
    //    val testDF = dataframeCreatorController.getTestDataframe("src/main/resources/test.csv",ss,sc)

    /**Necessary Dataframes creation
      * We create them using the class dataframeCreatorController
      * Necessary params: the file path which corresponds to the file for which we create the DF
      * !!!!The files need to be inside resources folder where stopwords.csv is!!!!
      *
      */
    val attributesDF = dataframeCreatorController.getAttributesDataframe("src/main/resources/attributes.csv", ss, sc)
    val trainingDF = dataframeCreatorController.getTrainDataframe("src/main/resources/train.csv", ss, sc)
    val descriptionDF = dataframeCreatorController.getProductDescriptionDataframe("src/main/resources/product_descriptions.csv", ss, sc)
    val searchDF = dataframeCreatorController.getSearchTermDF("src/main/resources/train.csv", ss, sc)

    /**We perform a union between the attributesDF and trainingDF ~ thus combining the first two Dataframes
      **/
    val joinedTextsTrainingAndTestDF = dataframeCreatorController.uniteTwoDataframeTexts(trainingDF, attributesDF, ss, sc)

    /** joinedDescAndTestTrainDF contains a dataframe of schema :
      * root
      * |-- rID: string (nullable = true)
      * |-- rProductUID: string (nullable = true)
      * |-- rFilteredWords: array (nullable = true)
      * |    |-- element: string (containsNull = true)
      * |-- rRelevance: string (nullable = true)
      *
      * At the training process of ml algorithms we will need rFilteredWords, and rRelevance
      *
      * TODO If we find a faster way of executing the following method.. much appreciated
      */

    /**We also unite the joinedTextsTrainingAndTestDF with the descriptionDF
      * so that we get all the information in one place ~ of course after all the necessary preprocessing we performed
      * The Default without the similarities DF
      */
    val joinedDescAndTestTrainDF = dataframeCreatorController.uniteTwoDataframeTexts(joinedTextsTrainingAndTestDF, descriptionDF, ss, sc)

    /**Similarities Section - use them only if you want to calculate the results with the similarities */

    /**Declare similarity methods object - uncomment to use*/
//    val similarity = new SimilarityMethods()

   /**This is for MinHash Similarity - uncomment to use*/
//    val jaccardSimilarityDF = similarity.calcJacccardSimilarity(searchDF,joinedDescAndTestTrainDF,dataframeCreatorController,ss,sc)

    /**This is for CosineSimilarity with Row Matrix - uncomment to use*/
//    val cosineSimilarityDF = similarity.calcCosineSimilarity(searchDF,joinedDescAndTestTrainDF,dataframeCreatorController,ss,sc)
//
//    /**This is for Euclidean Similarity - uncomment to use*/
//    val euclideanSimilarityDF = similarity.calcEuclideanSimilarity(searchDF,joinedDescAndTestTrainDF,dataframeCreatorController,ss,sc)
//

    // val joinedDescAndTestTrainCount = joinedDescAndTestTrainDF.count()
//
//       joinedTextsTrainingAndTestDF.take(5).foreach(println)
//

    /**IMPORTANT: In order to calculate the results of the ML Models
      * we need to prepare the data for their final journey.
      * This means that we need to have the data in a LabelPoints format
      * which means also to have labels and features
      * ---------------------------------------------
      *
      * IMPORTANT: The next ML Models are used without the similarity
      * To see the results for each one you need to :
      * 1)comment out the other models of the one we want to use
      * 2)Uncomment the idf and trainingDfDataInit creation
      * 3)If already used delete the previous trainLabeledVectors folder from /src/main/resources
      * 3)If already used delete the previous Machine Learning  folder which resides in /target/tmp folder
      */
    //    ////TODO creation of labeledVectorTrain file. After first time creation, we just need to load it
//    //    val idf = dataframeCreatorController.getIDF(trainingDF,ss,sc)
//    val idf = dataframeCreatorController.getIDF(joinedDescAndTestTrainDF, ss, sc)
//    val trainingDfDataInit = idf.select($"rProductUID", $"rFilteredWords", $"rFeatures", $"rRelevance", $"rrLabel").withColumnRenamed("rFeatures", "features").withColumnRenamed("rrLabel", "label")
////
//    //    /* ======================================================= */
////    /* ================== REGRESSION ===================== */
////    /* ================== Decision Tree ====================*/
////    /* ======================================================= */
////
//    println("Defining features and label for the model...")
//    val trainingDfData = trainingDfDataInit.select("label","features")
//
//
//    val decisionTree_Regressor = new DecisionTree_Regressor()
//    decisionTree_Regressor.runPrediction(trainingDfData, sc, ss)

//
    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Random Forest ================== */
    /* ======================================================= */
//    println("Defining features and label for the model...")
//    val trainingDfData = trainingDfDataInit.select("features", "label")
//    val random_forest_Regressor = new RandomForest_Regressor()
//    random_forest_Regressor.runPrediction(trainingDfData,sc,ss)

    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Linear Regression ================== */
    /* ======================================================= */
//    println("Defining features and label for the model...")
//    val trainingDfData = trainingDfDataInit.select("features", "label")
//    val linearRegression = new Linear_Regressor()
//    linearRegression.runPrediction(trainingDfData,sc,ss)

    /**---------------------------------------------------------------------------------------------------------*/


    /**IMPORTANT: In order to calculate the results of the ML Models
      * we need to prepare the data for their final journey.
      * This means that we need to have the data in a LabelPoints format
      * which means also to have labels and features
      * ---------------------------------------------
      *
      * IMPORTANT: The next ML Models are used with the similarity with a limit of 50000 as a sample
      * because of the difficult of the procedure in order to train the model(highly difficult for the pc to handle)
      * To see the results for each one you need to :
      * 1)comment out the other models of the one we want to use
      * 2)Uncomment trainingDfDataInit creation
      * 3)If already used delete the previous trainLabeledVectors folder from /src/main/resources
      * 3)If already used delete the previous Machine Learning folder which resides in /target/tmp folder
      */
//


    /**Jaccard Similarity(Distance)*/
//    val trainingDfDataInit = jaccardSimilarityDF.select("Jaccard Similarity","rrLabel").withColumnRenamed("Jaccard Similarity","features")
//      .withColumnRenamed("rrLabel","label")

        /* ======================================================= */
        /* ================== REGRESSION ===================== */
        /* ================== Decision Tree ====================*/
        /* ======================================================= */

//        println("Defining features and label for the model...")
//        val trainingDfData = trainingDfDataInit.select("label", "features").limit(50000)
//        trainingDfDataInit.printSchema()
//
//        val decisionTree_Regressor = new DecisionTree_Regressor()
//
//        decisionTree_Regressor.runPrediction(trainingDfData, sc, ss)
//

    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Random Forest ================== */
    /* ======================================================= */
//        println("Defining features and label for the model...")
//        val trainingDfData = trainingDfDataInit.select("features", "label").limit(50000)
//        val random_forest_Regressor = new RandomForest_Regressor()
//        random_forest_Regressor.runPrediction(trainingDfData,sc,ss)

    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Linear Regression ================== */
    /* ======================================================= */
//        println("Defining features and label for the model...")
//        val trainingDfData = trainingDfDataInit.select("features", "label").limit(50000)
//        val linearRegression = new Linear_Regressor()
//        linearRegression.runPrediction(trainingDfData,sc,ss)

    /**---------------------------------------------------------------------------------------------------------*/
    /**Cosine Similarity*/

//    val trainingDfDataInit = cosineSimilarityDF.select("Similarity","rrLabel").withColumnRenamed("Similarity","features")
//          .withColumnRenamed("rrLabel","label")
    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Decision Tree ====================*/
    /* ======================================================= */

//    println("Defining features and label for the model...")
    //        val trainingDfData = trainingDfDataInit.select("label", "features").limit(50000)
    //        trainingDfDataInit.printSchema()
    //
    //        val decisionTree_Regressor = new DecisionTree_Regressor()
    //
    //        decisionTree_Regressor.runPrediction(trainingDfData, sc, ss)


    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Random Forest ================== */
    /* ======================================================= */
    //        println("Defining features and label for the model...")
    //        val trainingDfData = trainingDfDataInit.select("features", "label").limit(50000)
    //        val random_forest_Regressor = new RandomForest_Regressor()
    //        random_forest_Regressor.runPrediction(trainingDfData,sc,ss)

    /**---------------------------------------------------------------------------------------------------------*/
    /**Euclidean Similarity(Distance)*/

//    val trainingDfDataInit = euclideanSimilarityDF.select("Euclidean Similarity","rrLabel").withColumnRenamed("Euclidean Similarity","features")
    //          .withColumnRenamed("rrLabel","label")
    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Decision Tree ====================*/
    /* ======================================================= */

    //    println("Defining features and label for the model...")
    //        val trainingDfData = trainingDfDataInit.select("label", "features").limit(50000)
    //        trainingDfDataInit.printSchema()
    //
    //        val decisionTree_Regressor = new DecisionTree_Regressor()
    //
    //        decisionTree_Regressor.runPrediction(trainingDfData, sc, ss)

    /* ======================================================= */
    /* ================== REGRESSION ===================== */
    /* ================== Random Forest ================== */
    /* ======================================================= */
    //        println("Defining features and label for the model...")
    //        val trainingDfData = trainingDfDataInit.select("features", "label").limit(50000)
    //        val random_forest_Regressor = new RandomForest_Regressor()
    //        random_forest_Regressor.runPrediction(trainingDfData,sc,ss)


    sc.stop()
  }



}
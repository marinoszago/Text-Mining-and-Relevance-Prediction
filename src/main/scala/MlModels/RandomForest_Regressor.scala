package MlModels

import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, SparkSession}

class RandomForest_Regressor extends Serializable {

  def runPrediction(trainingDfData : DataFrame, sc : SparkContext, ss : SparkSession): Unit ={
    createLabelPoints(trainingDfData,ss)
    val data = MLUtils.loadLibSVMFile(sc, "src/main/resources/trainLabeledVectors.csv")

    /** Creation of ML Model, as Labeled file, we can create it, save it to memory, for future uses */

    val splits = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val (trainingData, testData) = (splits(0), splits(1))
    trainingData.cache()
    testData.cache()

    var modelExist = Files.exists(Paths.get("target/tmp/myRandomForestRegressionModel"))

    if (!modelExist) {

      println("Applying Random Forest regression...")

      val numClasses = 2
      val categoricalFeaturesInfo = Map[Int, Int]()
      val numTrees = 5 // 100 in the whole dataset, we don't have the memory, 10 here because I have 400 examples
      val featureSubsetStrategy = "auto" // Let the algorithm choose.
      val impurity = "variance"
      val maxDepth = 5
      val maxBins = 32

      val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
        numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

      model.save(sc, "target/tmp/myRandomForestRegressionModel")

    }

    val model = RandomForestModel.load(sc, "target/tmp/myRandomForestRegressionModel")

    /** Evaluation of RandomForest model on test instances and compute test error */
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
    println("Learned regression random forest model:\n" + model.toDebugString)
    println("Test Mean Squared Error = " + testMSE)

  }


  def createLabelPoints(trainingDfData: DataFrame, ss : SparkSession): Unit ={
    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    var labeledPointsExist = Files.exists(Paths.get("src/main/resources/trainLabeledVectors.csv"))

    if (!labeledPointsExist) {
      /** TODO COMMENT the creation of labeled and convertedVecDF if we already have saved the trainLabeledVectors.csv in resources !*/
      /** Creating Labeled data out of training Dataframe because that is the form ml algorithms accept them*/
      val labeled = trainingDfData.rdd.map(row => LabeledPoint(
        row.getAs[Double]("label"),
        row.getAs[org.apache.spark.ml.linalg.Vector]("features")
      )).toDS()

      /** Convert this labels into ML and save it to resources for later use,
        * If we already have that file into resources we dont need again to re create it.
        * Thus we save a lot of computational time*/
      val convertedVecDF = MLUtils.convertVectorColumnsToML(labeled)
      convertedVecDF.write.format("libsvm").save("src/main/resources/trainLabeledVectors.csv")

    }

  }




}

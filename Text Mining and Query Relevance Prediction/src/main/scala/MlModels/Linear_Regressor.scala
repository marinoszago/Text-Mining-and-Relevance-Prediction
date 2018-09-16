package MlModels

import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.optimization.SimpleUpdater

class Linear_Regressor extends Serializable {

  def runPrediction(trainingDfData : DataFrame, sc : SparkContext, ss : SparkSession): Unit ={
    createLabelPoints(trainingDfData,ss)
    val data = MLUtils.loadLibSVMFile(sc, "src/main/resources/trainLabeledVectors.csv")

    /** Creation of ML Model, as Labeled file, we can create it, save it to memory, for future uses */

    val splits = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val (trainingData, testData) = (splits(0), splits(1))

    trainingData.cache()
    testData.cache()


    println("Applying Linear regression...")

    val model = LinearRegressionWithSGD.train(trainingData, 100, 1.0)

    /** Evaluation of Linear Regression on test instances and compute test error */
    val labelsAndPredictions = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testMSE = labelsAndPredictions.map{ case (v, p) => math.pow(v - p, 2) }.mean()
//    labelsAndPredictions.take(10).foreach(println)
    //Print the coefficients and intercept for linear regression
    println("Test Mean Squared Error = " + testMSE)
    sc.stop()

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
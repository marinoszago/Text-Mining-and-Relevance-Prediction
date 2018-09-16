package TextProcessingController

import breeze.linalg.{SparseVector, norm}
import com.sun.prism.PixelFormat.DataType
import org.apache.spark.{SparkContext, rdd}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, MatrixEntry, RowMatrix}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, monotonically_increasing_id, udf}



class SimilarityMethods() extends Serializable {

  def calcCosineSimilarity(searchDataframe: DataFrame, featuresDataframe: DataFrame, dataframeCreatorController: DataframeCreatorController, ss: SparkSession, sc: SparkContext): DataFrame = {


    /**Get the TF-IDF for featuresDataframe and searchDataframe in order to have the necessary vectors*/

    val idf1 = dataframeCreatorController.getIDF(featuresDataframe, ss, sc)
    val idf2 = dataframeCreatorController.getSearchIDF(searchDataframe, ss, sc)

    /**Normalize the columns necessary for calculating Similarity in order to have more consistent results*/

    val norm1 = new Normalizer()
      .setInputCol("rFeatures")
      .setOutputCol("NormalizedFeatures")

    val normalizedFeatures = norm1.transform(idf1)
    normalizedFeatures.printSchema()

    val norm2 = new Normalizer()
      .setInputCol("rSearchFeatures")
      .setOutputCol("NormalizedSearchFeatures")
    val normalizedSearchFeaturesDF = norm2.transform(idf2)
    normalizedSearchFeaturesDF.printSchema()


    /**Perform the join and get all features in one Dataframe consisting of the normalized features*/

    val featuresAndSearchFeaturesDF = normalizedFeatures.join(normalizedSearchFeaturesDF, "rProductUID").select("rProductUID","NormalizedFeatures", "NormalizedSearchFeatures", "rrLabel")

    /**Create a vector using Vector Assembler from the two normalized columns and name it Similarity*/

    val assembler = new VectorAssembler()
      .setInputCols(Array("NormalizedFeatures", "NormalizedSearchFeatures"))
      .setOutputCol("Similarity")

    /**output: rrLabel
      *        Similarity
      */
        val output = assembler.transform(featuresAndSearchFeaturesDF).select("Similarity").rdd

    /**In order to calculate the Similarity between two columns we need to create the Matrix
      * To do that we get the first field of output rdd which contains the Similarity
      * Afterwards create a RowMatrix with the recently created vector
      * We can compute similarities either with a Brute force(simsPerfect) approach or
      * with an approximation(simsEstimate)
      *
      */
        val items_mllib_vector = output.map(_.getAs[org.apache.spark.ml.linalg.Vector](0))
          .map(org.apache.spark.mllib.linalg.Vectors.fromML)


        val mat = new RowMatrix(items_mllib_vector)
      /*Brute force approach*/
//        val simsPerfect = mat.columnSimilarities()

      /*With approximation*/
        val simsEstimate = mat.columnSimilarities(0.1) //using DISUM


    /**We now prepare the data to create the new dataframe so that we can use it later on the ML Models*/

        val transformedRDD = simsEstimate.entries.map{case MatrixEntry(row: Long, col:Long, sim:Double) => Array(sim).mkString(",")}
        //Transform rdd[String] to rdd[Row]
        val rdd2 = transformedRDD.map(a => Row(a))

        // to DF
    /**By creating monotonically increasing ids and doing orderby  we ensure that the two dataframes will join properly*/
        val dfschema = StructType(Array(StructField("Similarity",StringType)))
        val rddToDF = ss.createDataFrame(rdd2,dfschema).select("Similarity").withColumn("rowID2",monotonically_increasing_id())

        val relevanceDF = featuresAndSearchFeaturesDF.select("rProductUID","rrLabel")
          .orderBy("rProductUID")
          .withColumn("rowID1",monotonically_increasing_id())

        val cosineDF = rddToDF.join(relevanceDF,rddToDF("rowID2")===relevanceDF("rowID1"),"inner")
              .select("rProductUID","Similarity","rrLabel").orderBy("rProductUID")

    cosineDF

  }



  def calcJacccardSimilarity(searchDataframe: DataFrame,featuresDataframe:DataFrame,dataframeCreatorController : DataframeCreatorController,ss : SparkSession,sc: SparkContext) : DataFrame = {
    /*First Find the TF using Hashing TF. CountVectorizer can also be used */
    val featuresTF = dataframeCreatorController.getTF(featuresDataframe,ss,sc).select("rID","rTFFeatures","rRelevance")

    val joinedDF = featuresTF.join(searchDataframe,"rID").select("rID","rProductUID","rSearchTermFeatures","rTFFeatures","rRelevance").orderBy("rID")

    /*Create MinHash object*/
    val lsh = new MinHashLSH().setInputCol("rTFFeatures").setOutputCol("LSH").setNumHashTables(3)

    /*Create Pipeline model*/
    val pipe = new Pipeline().setStages(Array(lsh))
    val pipeModel = pipe.fit(joinedDF)

    /*Create the new DF*/

    val transformedDF = pipeModel.transform(joinedDF)

    /*Create Transformer*/
    val transformer = pipeModel.stages

    /*MinHashModel*/
    val tempMinHashModel = transformer.last.asInstanceOf[MinHashLSHModel]
    val threshold = 1.5

    /*Just a udf for converting string to double*/
    val udf_toDouble = udf( (s: String) => s.toDouble )


    /*Perform the Similarity with self-join*/
    /*Find the distance of pairs which is lower than the given threshold*/

    val preSimilarityDF = tempMinHashModel.approxSimilarityJoin(transformedDF,transformedDF,threshold)
      .select(udf_toDouble(col("datasetA.rRelevance")).alias("rrLabel"),
          col("distCol"))

    /*Make a vector of the distCol and name it Similarity. It will be needed when using the df for the ML Models*/
    val vectorAssem = new VectorAssembler()
      .setInputCols(Array("distCol"))
      .setOutputCol("Similarity")


    val jaccardSimilarityDF = vectorAssem.transform(preSimilarityDF).select("Similarity","rrLabel").withColumnRenamed("Similarity","Jaccard Similarity")



    jaccardSimilarityDF
  }

  def calcEuclideanSimilarity(searchDataframe: DataFrame,featuresDataframe:DataFrame,dataframeCreatorController : DataframeCreatorController,ss : SparkSession,sc: SparkContext) : DataFrame ={
    /*First Find the TF using Hashing TF. CountVectorizer can also be used */
    val featuresTF = dataframeCreatorController.getTF(featuresDataframe,ss,sc).select("rID","rTFFeatures","rRelevance")
    val joinedDF = featuresTF.join(searchDataframe,"rID").select("rID","rSearchTermFeatures","rTFFeatures","rRelevance").orderBy("rID")

    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(2.0)
      .setNumHashTables(3)
      .setInputCol("rTFFeatures")
      .setOutputCol("BRPHashes")

    /*Create Pipeline model*/
    val pipe = new Pipeline().setStages(Array(brp))
    val pipeModel = pipe.fit(featuresTF)

    /*Transform the Dataframe*/
    val transformedDF = pipeModel.transform(joinedDF)

    /*Create Transformer*/
    val transformer = pipeModel.stages
    /*MinHashModel*/
    val tempMinHashModel = transformer.last.asInstanceOf[BucketedRandomProjectionLSHModel]

    /*Threshold*/
    val threshold = 1.5

    /*Just a udf for converting string to double*/
    val udf_toDouble = udf( (s: String) => s.toDouble )

    val preEuclideanSimilarityDF = tempMinHashModel.approxSimilarityJoin(transformedDF,transformedDF,threshold,"distCol")
      .select(udf_toDouble(col("datasetA.rRelevance")).alias("rrLabel"),
        col("distCol"))

    /*Make a vector of the distCol and name it Similarity. It will be needed when using the df for the ML Models*/
    val vectorAssem2 = new VectorAssembler()
      .setInputCols(Array("distCol"))
      .setOutputCol("Similarity")

    val euclideanSimilarityDF = vectorAssem2.transform(preEuclideanSimilarityDF).select("Similarity","rrLabel").withColumnRenamed("Similarity","Euclidean Similarity")

    euclideanSimilarityDF
  }


}



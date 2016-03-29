package org.apache.spark.examples.kaggle

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder

/*
 * Mainly follow the pipeline proposed by Manisha S.
 * https://developer.ibm.com/spark/blog/2016/02/22/predictive-model-for-online-advertising-using-spark-machine-learning-pipelines/
 */
object CriteoCtrPrediction {
  private val schema = StructType(Array(
    StructField("Label", DoubleType, false),
    StructField("I1", IntegerType, true),
    StructField("I2", IntegerType, true),
    StructField("I3", IntegerType, true),
    StructField("I4", IntegerType, true),
    StructField("I5", IntegerType, true),
    StructField("I6", IntegerType, true),
    StructField("I7", IntegerType, true),
    StructField("I8", IntegerType, true),
    StructField("I9", IntegerType, true),
    StructField("I10", IntegerType, true),
    StructField("I11", IntegerType, true),
    StructField("I12", IntegerType, true),
    StructField("I13", IntegerType, true),
    StructField("C1", StringType, true),
    StructField("C2", StringType, true),
    StructField("C3", StringType, true),
    StructField("C4", StringType, true),
    StructField("C5", StringType, true),
    StructField("C6", StringType, true),
    StructField("C7", StringType, true),
    StructField("C8", StringType, true),
    StructField("C9", StringType, true),
    StructField("C10", StringType, true),
    StructField("C11", StringType, true),
    StructField("C12", StringType, true),
    StructField("C13", StringType, true),
    StructField("C14", StringType, true),
    StructField("C15", StringType, true),
    StructField("C16", StringType, true),
    StructField("C17", StringType, true),
    StructField("C18", StringType, true),
    StructField("C19", StringType, true),
    StructField("C20", StringType, true),
    StructField("C21", StringType, true),
    StructField("C22", StringType, true),
    StructField("C23", StringType, true),
    StructField("C24", StringType, true),
    StructField("C25", StringType, true),
    StructField("C26", StringType, true)))

  def toInt(s: String): Option[Int] = {
    try {
      Some(s.toInt)
    } catch {
      case e: Exception => None
    }
  }

  def parseData(data: RDD[String], sqlContext: SQLContext): DataFrame = {
    // Split the csv file by comma and convert each line to a tuple.
    val parts = data.map(line => line.split("\t", -1))
    parts.take(10).foreach(arr => println(arr.size))
    val features = parts.map(p => Row(p(0).toDouble, toInt(p(1)), toInt(p(2)), toInt(p(3)), toInt(p(4)), toInt(p(5)),
      toInt(p(6)), toInt(p(7)), toInt(p(8)), toInt(p(9)), toInt(p(10)), toInt(p(11)), toInt(p(12)), toInt(p(13)),
      p(14), p(15), p(16), p(17), p(18), p(19),
      p(20), p(21), p(22), p(23), p(24), p(25), p(26), p(27), p(28), p(29),
      p(30), p(31), p(32), p(33), p(34), p(35), p(36), p(37), p(38), p(39)))

    // Apply the schema to the RDD.
    return sqlContext.createDataFrame(features, schema)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("CriteoCtrPrediction")
    val sc = new SparkContext(conf)

    // $example on$
//    val sampleData = sc.textFile("D:\\Datasets\\Criteo\\dac_sample\\dac_sample.txt.mini", 2)
    val sampleData = sc.textFile("D:\\Datasets\\Criteo\\dac_sample\\dac_sample.txt", 2)

    println(s"Data size is ${sampleData.count}")
    sampleData.take(2).foreach(println)

    val sqlContext = new SQLContext(sc)
    // Register the DataFrame as a table.
    val schemaClicks = parseData(sampleData, sqlContext)
    schemaClicks.registerTempTable("clicks")
    schemaClicks.printSchema()

    val cols = Seq("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26")
    // Replace empty values in Categorical features by "NA"
    val schemaClicksNA = schemaClicks.na.replace(cols, Map("" -> "NA"))
    // Drop rows containing null values in the DataFrame
    val schemaClicksCleaned = schemaClicksNA.na.drop()

    val Array(trainData, testData) = schemaClicksCleaned.randomSplit(Array(0.9, 0.1), seed = 42)
    trainData.cache()
    testData.cache()

    //    val indexer = new StringIndexer()
    //      .setInputCol("category")
    //      .setOutputCol("categoryIndex")
    //
    //    val encoder = new OneHotEncoder()
    //      .setInputCol("categoryIndex")
    //      .setOutputCol("categoryVec")
    //      
    //    val assembler = new VectorAssembler()
    //      .setInputCols(Array("hour", "mobile", "userFeatures"))
    //      .setOutputCol("features")
    //      
    //    val pipeline = new Pipeline()
    //      .setStages(Array(indexer, encoder, assembler))
    //
    //    // Fit the pipeline to training documents.
    //    val model = pipeline.fit(trainData)

    // Union data for one-hot encoding
    // To extract features throughly, union the training and test data.
    // Since the test data includes values which doesn't exists in the training data.
    val train4union = trainData.select(cols.map(col): _*)
    val test4union = testData.select(cols.map(col): _*)
    val union = train4union.unionAll(test4union).cache()

    // Extracts features with one-hot encoding
    def getIndexedColumn(column: String): String = s"${column}_indexed"
    def getColumnVec(column: String): String = s"${column}_vec"
    val feStages = ArrayBuffer.empty[PipelineStage]
    cols.foreach { clm =>
      val stringIndexer = new StringIndexer()
        .setInputCol(clm)
        .setOutputCol(getIndexedColumn(clm))
        .setHandleInvalid("error")
      val oneHotEncoder = new OneHotEncoder()
        .setInputCol(getIndexedColumn(clm))
        .setOutputCol(getColumnVec(clm))
        .setDropLast(false)
      Array(stringIndexer, oneHotEncoder)
      feStages.append(stringIndexer)
      feStages.append(oneHotEncoder)
    }
    val va = new VectorAssembler()
      .setInputCols(cols.toArray.map(getColumnVec))
      .setOutputCol("features")
    feStages.append(va)
    val fePipeline = new Pipeline().setStages(feStages.toArray)
    val feModel = fePipeline.fit(union)
    val trainDF = feModel.transform(trainData).select("Label", "features").cache()
    val testDF = feModel.transform(testData).select("Label", "features").cache()
    union.unpersist()
    trainData.unpersist()
    testData.unpersist()
    trainDF.show(5)
    testDF.show(5)
    
     val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("Label")
//      .setRegParam(0.3)
//      .setElasticNetParam(0.8)
//      .setMaxIter(10)
//      .setTol(1E-6)
//      .setFitIntercept(true)

    // Fit the Pipeline
    val startTime = System.nanoTime()
//    val lrModel = lr.fit(trainDF)
    
     val paramGrid = new ParamGridBuilder()
//      .addGrid(lr.regParam, Array(1, 0.3, 0.2, 0.1, 0.01, 0.001))
//      .addGrid(lr.elasticNetParam, Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
      .addGrid(lr.regParam, Array(1, 0.1, 0.01))
      .addGrid(lr.elasticNetParam, Array(0.2, 0.8))
      .build()

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("Label")
    val cv = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)  // Use 3+ in practice

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(trainDF)

    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $elapsedTime seconds")

    // Print the weights and intercept for logistic regression.
//    println(s"Weights: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    println("Training data results:")
    evaluateClassificationModel(cvModel, trainDF, "Label")
    println("Test data results:")
    evaluateClassificationModel(cvModel, testDF, "Label")

    // $example off$
    sc.stop()
  }
  
  /**
   * Evaluate the given ClassificationModel on data.  Print the results.
   * @param model  Must fit ClassificationModel abstraction
   * @param data  DataFrame with "prediction" and labelColName columns
   * @param labelColName  Name of the labelCol parameter for the model
   *
   * TODO: Change model type to ClassificationModel once that API is public. SPARK-5995
   */
  def evaluateClassificationModel(
      model: Transformer,
      data: DataFrame,
      labelColName: String): Unit = {
    val fullPredictions = model.transform(data).cache()
    val predictions = fullPredictions.select("prediction").rdd.map(_.getDouble(0))
    val labels = fullPredictions.select(labelColName).rdd.map(_.getDouble(0))
    // Print number of classes for reference
//    val numClasses = MetadataUtils.getNumClasses(fullPredictions.schema(labelColName)) match {
//      case Some(n) => n
//      case None => throw new RuntimeException(
//        "Unknown failure when indexing labels for classification.")
//    }
    val numClasses = 2
    val accuracy = new MulticlassMetrics(predictions.zip(labels)).precision
    println(s"  Accuracy ($numClasses classes): $accuracy")
  }
}
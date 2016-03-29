package org.apache.spark.examples.kaggle

/**
 * https://issues.apache.org/jira/browse/SPARK-10935
 * 
 * Code copied from:
 * https://github.com/yinxusen/incubator-project/blob/b332de87606b4599d96a6cc41aadb934f2f30577/avito/src/main/scala/org/apache/spark/examples/main.scala
 * 
 * Other useful repos:
 * https://github.com/bluebytes60/SparkML/tree/master/src/main/scala/avito
 * https://github.com/Sirorezka/SNA_Hackaton/blob/a03db56b329fdea116febfeb946e2db6f33ae610/My_Model_scala/src/main/scala/Baseline.scala
 */
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{ Row, SQLContext }
import org.apache.spark.{ SparkConf, SparkContext }

case class SearchStream(
  searchId: Int,
  adId: Int,
  position: Int,
  objectType: Int,
  histCTR: Double,
  isClick: Double)

object AvitoContextAdClicks {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[4]").setAppName("Avito")
    val sc = new SparkContext(conf)
    val sqlCtx = new SQLContext(sc)
    import sqlCtx.implicits._

    // here we need lots of work to load data
    val trainSearchStream = sc.textFile("/Users/panda/data/ads/trainSearchStream.tsv")
      .map(_.split('\t'))
      .filter(!_.contains("SearchID"))
      .filter(_.length == 6)
      .map { record =>
        SearchStream(
          record(0).toInt,
          record(1).toInt,
          record(2).toInt,
          record(3).toInt,
          record(4).toDouble,
          record(5).toDouble)
      }.toDF()

    val valueForClicks = trainSearchStream.select("isClick").distinct()
      .map { case Row(isClick: Double) => isClick }.collect()

    val assembler = new VectorAssembler()
      .setInputCols(Array("adId", "position", "objectType", "histCTR")).setOutputCol("feature")

    val dataSet = assembler.transform(trainSearchStream).select("feature", "isClick")

    val splits = dataSet.randomSplit(Array(0.7, 0.3))
    val trainingSet = splits(0)
    val testSet = splits(1)

    val lr = new LogisticRegression()
      .setMaxIter(20)
      .setRegParam(0.03)
      .setElasticNetParam(0.1)
      .setFeaturesCol("feature")
      .setLabelCol("isClick")
      .setRawPredictionCol("result")

    val lrModel = lr.fit(trainingSet)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("isClick").setPredictionCol("result")

    val eval = evaluator.evaluate(lrModel.transform(testSet))
    println(eval)
  }
}
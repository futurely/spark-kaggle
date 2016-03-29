package org.apache.spark.examples.kaggle

import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
 * https://issues.apache.org/jira/browse/SPARK-10513
 * Copied from:
 * https://github.com/yanboliang/Springleaf
 *
 * Created by yanboliang on 9/30/15.
 */
object Springleaf {

  val trainFile = "src/main/resources/kaggle/small-train.csv"
  val testFile = "src/main/resources/kaggle/test.csv"

  def main(args: Array[String]): Unit = {

    println("Springleaf start")

    val sc = new SparkContext("local", "Springleaf")
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val training = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(trainFile)

    val numericColumnNames = training.schema.fields.filter(_.dataType != StringType).map(_.name).filter(_ != "target").toSeq

    val categoryColumnIndex = Seq("0001", "0005", "0008", "0009", "0010", "0011", "0012", "0043", "0196", "0200", 
        "0202", "0216", "0222", "0226", "0229", "0230", "0232", "0236", "0237", "0239" 
        /*, "0274", "0283", "0305", "0325", "0342", "0353", "0467"*/ )

    val categoryColumnNames = categoryColumnIndex.map("VAR_" + _)

    val allFeatureColumns = numericColumnNames ++ categoryColumnNames

    val training2 = training.select("target", allFeatureColumns: _*)
    //training2.show()

    var oldTraining: DataFrame = training2
    var newTraining: DataFrame = training2

    categoryColumnIndex.foreach {
      x =>
        {
          val colName = "VAR_" + x
          //println(colName)
          val indexer = new StringIndexer()
            .setInputCol(colName)
            .setOutputCol(colName + "_indexed")
            .fit(oldTraining)
          val indexed = indexer.transform(oldTraining)

          val encoder = new OneHotEncoder()
            .setDropLast(false)
            .setInputCol(colName + "_indexed")
            .setOutputCol(colName + "_encoded")
          newTraining = encoder.transform(indexed)

          oldTraining = newTraining
        }
    }

    val assemblerNames = (numericColumnNames ++ categoryColumnNames.map(_ + "_encoded")).filter(_ != "ID").filter(_ != "target").toArray
    //println(assemblerNames.mkString(","))

    val assembler = new VectorAssembler()
      .setInputCols(assemblerNames)
      .setOutputCol("features")

    val training3 = assembler.transform(newTraining)
    val training4 = training3.withColumn("label", col("target").cast(DoubleType))

    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setLabelCol("label")
      .setFeaturesCol("features")

    val model = lr.fit(training4)
    model.transform(training4).select("ID", "label", "prediction").show(200, false)
  }
}
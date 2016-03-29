package org.apache.spark.examples.kaggle

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{ StringIndexerModel, StandardScaler, StringIndexer, VectorAssembler }
import org.apache.spark.sql.{ SQLContext, Row }
import org.apache.spark.ml.classification.{ DecisionTreeClassifier, LogisticRegression }

/**
 * https://issues.apache.org/jira/browse/SPARK-10055
 * Copied from:
 * https://github.com/Lewuathe/spark-kaggle-examples
 * 
 * Created by sasakikai on 8/24/15.
 */
object SfCrimeClassification {

  def labelToVec(label: Int, labels: Array[String], sortedLabels: Array[String]): Array[Int] = {
    require(labels.length == sortedLabels.length)
    val stringLabel = labels(label)
    val sortedIndex = sortedLabels.indexOf(stringLabel)
    val ret = new Array[Int](labels.length)
    ret(sortedIndex) = 1
    ret
  }

  def main(args: Array[String]) {
    if (args.length < 3) {
      println("File path must be passed. " + args.length)
      System.exit(-1)
    }
    val trainFilePath = args(0)
    val testFilePath = args(1)
    val outputFilePath = args(2)
    val conf = new SparkConf().setAppName("SfCrimeClassification")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    /**
     * Training Phase
     */
    val trainData = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true").option("inferSchema", "true").load(trainFilePath)

    val categoryIndexer = new StringIndexer().setInputCol("Category")
      .setOutputCol("label")
    val dayOfWeekIndexer = new StringIndexer().setInputCol("DayOfWeek")
      .setOutputCol("DayOfWeekIndex")
    val pdDistrictIndexer = new StringIndexer().setInputCol("PdDistrict")
      .setOutputCol("PdDistrictIndex")
    val vectorAssembler = new VectorAssembler().setInputCols(Array("DayOfWeekIndex",
      "PdDistrictIndex", "X", "Y")).setOutputCol("rowFeatures")
    val featureScaler = new StandardScaler().setInputCol("rowFeatures")
      .setOutputCol("features")
    val classifier = new DecisionTreeClassifier()

    val trainPipeline = new Pipeline().setStages(Array(categoryIndexer, dayOfWeekIndexer,
      pdDistrictIndexer, vectorAssembler, featureScaler, classifier))

    val model = trainPipeline.fit(trainData)

    /**
     * Test Phase
     */
    val testData = sqlContext.read.format("com.databricks.spark.csv")
      .option("header", "true").option("inferSchema", "true").load(testFilePath)

    val labels = Array("LARCENY/THEFT", "OTHER OFFENSES", "NON-CRIMINAL", "ASSAULT", "DRUG/NARCOTIC",
      "VEHICLE THEFT", "VANDALISM", "WARRANTS", "BURGLARY", "SUSPICIOUS OCC", "MISSING PERSON",
      "ROBBERY", "FRAUD", "FORGERY/COUNTERFEITING", "SECONDARY CODES", "WEAPON LAWS", "PROSTITUTION",
      "TRESPASS", "STOLEN PROPERTY", "SEX OFFENSES FORCIBLE", "DISORDERLY CONDUCT", "DRUNKENNESS",
      "RECOVERED VEHICLE", "KIDNAPPING", "DRIVING UNDER THE INFLUENCE", "RUNAWAY", "LIQUOR LAWS",
      "ARSON", "LOITERING", "EMBEZZLEMENT", "SUICIDE", "FAMILY OFFENSES", "BAD CHECKS", "BRIBERY",
      "EXTORTION", "SEX OFFENSES NON FORCIBLE", "GAMBLING", "PORNOGRAPHY/OBSCENE MAT", "TREA")

    val writer = new PrintWriter(new File(outputFilePath))
    writer.write("Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS\n")
    model.transform(testData).select("Id", "prediction").collect().foreach {
      case Row(id: Int, prediction: Double) => {
        val labelVec = labelToVec(prediction.toInt, labels, labels.sortWith((s1, s2) => s1 < s2))
        writer.write(s"$id,${labelVec.mkString(",")}\n")
      }
    }
    writer.close()

  }
}
package estimators.builders

import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import estimators.Builder
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

class GBT extends Builder {

  private val inputCol = "GBT-features"
  private val outputCol = "GBT-prediction"

  override def createFeatures(data:DataFrame):DataFrame = {
    val toDenseVector = udf(
      (pclass:Double, sex:Double, age:Double, numberOfCabin:Double, embarkedClass:Double, isAlone:Double, title:Double) =>
      Vectors.dense(Array(pclass, sex, age, numberOfCabin, embarkedClass, isAlone, title))
    )
    data.drop("sibsp", "sibsp", "parch", "fare", "cabin", "family")
    data.withColumn(
      inputCol,
      toDenseVector(
        col("pclass"),
        col("sex"),
        col("age"),
        col("numberOfCabin"),
        col("embarkedClass"),
        col("isAlone"),
        col("title")
      )
    )
  }

  override def build(last:Boolean): PipelineStage = {
    // Create gradient boosting tree model.
    val gbt = new GBTClassifier()
      .setMaxIter(100)
      .setFeaturesCol(inputCol)
      .setPredictionCol(if (last) "prediction" else outputCol)

    // Create parameters grid.
    val paramGrid = new ParamGridBuilder()
      .addGrid(gbt.maxBins, Array(10))
      .addGrid(gbt.maxDepth, Array(3))
      .build()

    // Create trainer using validation split to evaluate which set of parameters performs the best.
    new TrainValidationSplit()
      .setEstimator(gbt)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8) // 80% of the data will be used for training and the remaining 20% for validation.
  }
}

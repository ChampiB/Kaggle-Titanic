package estimators.builders

import estimators.Builder
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

class LogisticClassifier extends Builder {

  private val inputCol = "LogisticClassifier-features"
  private val outputCol = "LogisticClassifier-prediction"

  override def createFeatures(data: DataFrame): DataFrame = {
    val toDenseVector = udf((pclass:Double, sex:Double, age:Double, embarkedClass:Double) => Vectors.dense(
      Array(pclass, sex, age, embarkedClass)
    ))
    data.withColumn(
      inputCol,
      toDenseVector(col("pclass"), col("sex"), col("age"), col("embarkedClass"))
    )
  }

  override def build(last:Boolean): PipelineStage = {
    // Create gradient boosting tree model.
    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setFeaturesCol(inputCol)
      .setPredictionCol(if (last) "prediction" else outputCol)

    // Create parameters grid.
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.standardization, Array(true, false))
      .addGrid(lr.elasticNetParam, Array(0.25D, 0.5D, 0.75D))
      .build()

    // Create trainer using validation split to evaluate which set of parameters performs the best.
    new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8) // 80% of the data will be used for training and the remaining 20% for validation.
  }
}

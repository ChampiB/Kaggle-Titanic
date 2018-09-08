package estimators.builders

import estimators.Builder
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

class MLP() extends Builder {

  private val inputCol = "MLP-features"
  private val outputCol = "MLP-prediction"
  private val numberOfFeatures = 6

  override def createFeatures(data: DataFrame): DataFrame = {
    val toDenseVector = udf(
      (
        pclass:Double,
        sex:Double,
        age:Double,
        sibsp:Double,
        numberOfCabin:Double,
        embarkedClass:Double
      ) => Vectors.dense(Array(pclass, sex, age, sibsp, numberOfCabin, embarkedClass))
    )
    data.withColumn(
      inputCol,
      toDenseVector(
        col("pclass"),
        col("sex"),
        col("age"),
        col("sibsp"),
        col("numberOfCabin"),
        col("embarkedClass")
      )
    )
  }

  override def build(last:Boolean): PipelineStage = {
    // Create gradient boosting tree model.
    val mlp = new MultilayerPerceptronClassifier()
      .setMaxIter(200)
      .setFeaturesCol(inputCol)
      .setPredictionCol(if (last) "prediction" else outputCol)

    // Create parameters grid.
    val paramGrid = new ParamGridBuilder()
      .addGrid(mlp.tol, Array(1E-6, 1E-8, 1E-12))
      .addGrid(mlp.layers, Array(
        Array(numberOfFeatures, 2, 2, 2, 2, 2),
        Array(numberOfFeatures, 3, 3, 3, 3, 2),
        Array(numberOfFeatures, 4, 4, 4, 4, 2),
        Array(numberOfFeatures, 5, 5, 5, 5, 2)
      ))
      .build()

    // Create trainer using validation split to evaluate which set of parameters performs the best.
    new TrainValidationSplit()
      .setEstimator(mlp)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8) // 80% of the data will be used for training and the remaining 20% for validation.
  }
}

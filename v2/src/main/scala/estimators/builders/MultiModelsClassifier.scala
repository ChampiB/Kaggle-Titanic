package estimators.builders

import config.Paths
import estimators.{Builder, FeaturesCreator}
import org.apache.spark.ml.{PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

class MultiModelsClassifier extends Builder {

  private val inputCol = "MultiModelsClassifier-features"
  private val outputCol = "MultiModelsClassifier-prediction"
  private val numberOfFeatures = 3

  override def createFeatures(data: DataFrame): DataFrame = {
    // Create input data
    var _data = data

    _data = FeaturesCreator.create("GBT", _data)
    _data = FeaturesCreator.create("MLP", _data)
    _data = FeaturesCreator.create("Logistic", _data)

    _data = PipelineModel
      .load(s"${Paths.resourcesBasePath}/GBT")
      .transform(_data)
      .withColumnRenamed("prediction", "GBT-prediction")

    _data = PipelineModel
      .load(s"${Paths.resourcesBasePath}/MLP")
      .transform(_data)
      .withColumnRenamed("prediction", "MLP-prediction")

    _data = PipelineModel
      .load(s"${Paths.resourcesBasePath}/Logistic")
      .transform(_data)
      .withColumnRenamed("prediction", "LogisticClassifier-prediction")

    // Format data
    val toDenseVector = udf(
      (
        mlp:Double,
        gbt:Double,
        logistic:Double
      ) => Vectors.dense(Array(mlp, gbt, logistic))
    )

    _data.withColumn(
      inputCol,
      toDenseVector(
        col("MLP-prediction"),
        col("GBT-prediction"),
        col("LogisticClassifier-prediction")
      )
    )
  }

  override def build(last:Boolean): PipelineStage = {
    // Create gradient boosting tree model.
    val mlp = new MultilayerPerceptronClassifier()
      .setMaxIter(100)
      .setFeaturesCol(inputCol)
      .setPredictionCol(if (last) "prediction" else outputCol)

    // Create parameters grid.
    val paramGrid = new ParamGridBuilder()
      .addGrid(mlp.layers, Array(
        Array(numberOfFeatures, 6, 2),
        Array(numberOfFeatures, 6, 6, 2),
        Array(numberOfFeatures, 4, 2),
        Array(numberOfFeatures, 4, 4, 2)
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

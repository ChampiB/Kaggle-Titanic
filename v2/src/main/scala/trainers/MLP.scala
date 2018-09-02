package trainers

import helpers.SparkHelper
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

class MLP extends Trainer {
  def fit(data:DataFrame):TrainValidationSplitModel = {

    // Spark import
    val ss = SparkHelper.getSession
    import ss.implicits._

    // Create test and train set.
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 42)

    // Create gradient boosting tree model.
    val mlp = new MultilayerPerceptronClassifier().setMaxIter(200)

    // Create parameters grid.
    val first = training.take(1)(0).getAs[DenseVector]("features")
    val paramGrid = new ParamGridBuilder()
      .addGrid(mlp.layers, Array(
        Array(first.size, 5, 5, 5, 2),
        Array(first.size, 7, 7, 7, 2),
        Array(first.size, 9, 9, 9, 2),
        Array(first.size, 11, 11, 11, 2),
        Array(first.size, 13, 13, 13, 2),
      ))
      .build()

    // Create trainer using validation split to evaluate which set of parameters performs the best.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(mlp)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8) // 80% of the data will be used for training and the remaining 20% for validation.

    // Run validation training and choose the best set of parameters.
    val model = trainValidationSplit.fit(training)

    // Make predictions on test data.
    val predictionsAndLabels = model.transform(test)
      .map{row => (Math.round(row.getAs[Double]("prediction")).toDouble, row.getAs[Int]("label").toDouble)}
      .rdd

    // Instantiate metrics object.
    val metrics = new BinaryClassificationMetrics(predictionsAndLabels)

    // Get and print metrics.
    val total = predictionsAndLabels.count()
    val good = predictionsAndLabels.filter{case(prediction, label) => prediction == label}.count()

    val accuracy  = good.toDouble / total.toDouble
    val precision = metrics.precisionByThreshold.collect.toMap
    val recall    = metrics.recallByThreshold.collect.toMap
    val f1_score  = metrics.fMeasureByThreshold.collect.toMap

    println(s"Accuracy:  $accuracy")
    println(s"Precision: ${precision(1.0)}")
    println(s"Recall:    ${recall(1.0)}")
    println(s"F1-Score:  ${f1_score(1.0)}")

    // Return best model.
    model
  }
}

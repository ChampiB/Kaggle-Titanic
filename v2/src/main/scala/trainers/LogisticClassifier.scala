package trainers

import helpers.SparkHelper
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame

class LogisticClassifier extends Trainer {
  def fit(data:DataFrame):TrainValidationSplitModel = {

    // Spark import
    val ss = SparkHelper.getSession
    import ss.implicits._

    // Create test and train set.
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 42)

    // Create gradient boosting tree model.
    val lr = new LogisticRegression().setMaxIter(200)

    // Create parameters grid.
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.standardization, Array(true, false))
      .addGrid(lr.elasticNetParam, Array(0.25D, 0.5D, 0.75D))
      .build()

    // Create trainer using validation split to evaluate which set of parameters performs the best.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new BinaryClassificationEvaluator)
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
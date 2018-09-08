import estimators.{EstimatorFactory, FeaturesCreator}
import helpers.{LoaderHelper, LoggerHelper, SaverHelper, SparkHelper}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel

object Train extends App {
  // Avoid logging and start spark session
  println("[Stage 1]: Start spark")
  LoggerHelper.off()
  SparkHelper.startSession()

  // Load estimators
  println("[Stage 2]: Load estimators")
  val estimatorsName = args(0).split(",")
  var estimators = Array[PipelineStage]()
  for (i <- estimatorsName.indices)
    estimators ++= Array(EstimatorFactory.build(estimatorsName(i), i == estimatorsName.length - 1))

  // Create the pipeline
  println("[Stage 3]: Create pipeline")
  val pipeline = new Pipeline().setStages(estimators)

  // Load raw features
  println("[Stage 4]: Load raw features")
  val features = LoaderHelper.loadFeatures(config.Paths.trainSet)
  val labels   = LoaderHelper.loadLabel(config.Paths.trainSet)
  var data     = features.join(labels, "id")

  // Load custom features
  println("[Stage 5]: Load custom features")
  for (estimatorName <- estimatorsName)
    data = FeaturesCreator.create(estimatorName, data)
  data = data.persist(StorageLevel.MEMORY_AND_DISK_2)

  // Create test and train set.
  println("[Stage 6]: Create train and test sets")
  var Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 42)

  // Run validation training and choose the best set of parameters.
  println("[Stage 7]: Fit pipeline")
  training = training.persist(StorageLevel.MEMORY_ONLY_2)
  val model = pipeline.fit(training)
  training.unpersist()

  // Print the metrics
  println("[Stage 8]: Print metrics")
  printMetrics(test, model)

  // Unpersist data, save trained model and stop spark session
  println("[Stage 9]: Stop spark session")
  data.unpersist()
  SaverHelper.save(model, if (args.length >= 2) s"${config.Paths.resourcesBasePath}/${args(1)}" else config.Paths.model)
  SparkHelper.stopSession()

  def printMetrics(test:DataFrame, model:PipelineModel):Unit = {
    // Spark import
    val ss = SparkHelper.getSession
    import ss.implicits._

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
  }
}

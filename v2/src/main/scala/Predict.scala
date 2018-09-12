import estimators.FeaturesCreator
import helpers.{LoggerHelper, SaverHelper}
import org.apache.spark.ml.PipelineModel

object Predict extends App {
  if (args.length < 1)
    throw new Exception("HELP: ./binary modelName")
  LoggerHelper.off()
  // Load data
  var data = helpers.LoaderHelper.loadFeatures(config.Paths.testSet)
  data = FeaturesCreator.create(args(0), data)
  // Load model
  val modelPath = s"${config.Paths.resourcesBasePath}/${args(0)}"
  val model = PipelineModel.load(modelPath)
  // Make predictions and save them
  val predictions = model.transform(data)
  SaverHelper.save(predictions, config.Paths.result)
  println(s"Results saved in ${config.Paths.result}")
}

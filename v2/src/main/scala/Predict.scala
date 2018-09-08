import helpers.{LoggerHelper, SaverHelper}
import org.apache.spark.ml.PipelineModel

object Predict extends App {
  LoggerHelper.off()
  val data = helpers.LoaderHelper.loadFeatures(config.Paths.testSet)
  val modelPath = if (args.length >= 2) s"${config.Paths.resourcesBasePath}/${args(0)}" else config.Paths.model
  val model = PipelineModel.load(modelPath)
  val predictions = model.transform(data)
  SaverHelper.save(predictions, config.Paths.result)
  println(s"Results saved in ${config.Paths.result}")
}

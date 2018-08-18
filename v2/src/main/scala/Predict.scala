import helpers.{LoggerHelper, SaverHelper}
import org.apache.spark.ml.tuning.TrainValidationSplitModel

object Predict extends App {
  LoggerHelper.off()
  val data = helpers.LoaderHelper.loadFeatures(config.Paths.testSet)
  val model = TrainValidationSplitModel.load(config.Paths.model)
  val predictions = model.transform(data)
  SaverHelper.save(predictions, config.Paths.result)
  println(s"Results saved in ${config.Paths.result}")
}

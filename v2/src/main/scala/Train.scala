import java.io.File
import java.nio.file.{Files, Paths}

import helpers.{LoaderHelper, LoggerHelper, SparkHelper}
import org.apache.commons.io.FileUtils
import org.apache.spark.storage.StorageLevel
import trainers.{GBT, LogisticClassifier, MLP, Trainer}

object Train extends App {
  val modelMapping = Map[String, Trainer](
    "GBT" -> new GBT,
    "MLP" -> new MLP,
    "Logistic" -> new LogisticClassifier
  )
  LoggerHelper.off()
  SparkHelper.startSession()
  val features = LoaderHelper.loadFeatures(config.Paths.trainSet)
  val labels   = LoaderHelper.loadLabel(config.Paths.trainSet)
  val data     = features.join(labels, "id").persist(StorageLevel.MEMORY_ONLY_2)
  val model    = modelMapping(args(0)).fit(data)
  data.unpersist()
  if (Files.exists(Paths.get(config.Paths.model)))
    FileUtils.deleteDirectory(new File(config.Paths.model))
  model.save(config.Paths.model)
  SparkHelper.stopSession()
}

import java.io.File
import java.nio.file.{Files, Paths}

import helpers.{LoaderHelper, LoggerHelper, SparkHelper}
import org.apache.commons.io.FileUtils
import org.apache.spark.storage.StorageLevel
import trainers.GBT

object Train extends App {
  LoggerHelper.off()
  SparkHelper.startSession()
  val features = LoaderHelper.loadFeatures(config.Paths.trainSet)
  val labels   = LoaderHelper.loadLabel(config.Paths.trainSet)
  val data     = features.join(labels, "id").persist(StorageLevel.MEMORY_ONLY_2)
  val model    = GBT.fit(data)
  data.unpersist()
  if (Files.exists(Paths.get(config.Paths.model)))
    FileUtils.deleteDirectory(new File(config.Paths.model))
  model.save(config.Paths.model)
  SparkHelper.stopSession()
}

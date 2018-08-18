package helpers

import java.io.File
import java.nio.file.{Files, Paths}
import org.apache.commons.io.FileUtils
import org.apache.spark.sql.DataFrame

object SaverHelper {
  def save(df:DataFrame, path:String):Unit = {
    // Spark import
    val ss = SparkHelper.getSession
    import ss.implicits._
    // DataFrame export
    val prediction = df.select("id", "prediction")
      .map{row => (row.getAs[Double]("id").toInt, row.getAs[Double]("prediction").toInt)}
      .withColumnRenamed("_1", "PassengerId")
      .withColumnRenamed("_2", "Survived")
    if (Files.exists(Paths.get(path)))
      FileUtils.deleteDirectory(new File(path))
    prediction
      .coalesce(1)
      .write
      .option("header", "true")
      .csv(path)

  }
}

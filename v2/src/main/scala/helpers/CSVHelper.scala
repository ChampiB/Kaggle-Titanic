package helpers

import org.apache.spark.sql.DataFrame

object CSVHelper {

  /*
  ** Load csv with multi-line and quote
  */
  def load(file:String, separator:String = ","):DataFrame = {
    SparkHelper.getSession
      .read
      .option("header", "true")
      .option("sep", separator)
      .option("quote", "\"")
      .option("escape", "\"")
      .option("multiLine", "true")
      .option("inferSchema", "true")
      .csv(file)
  }
}
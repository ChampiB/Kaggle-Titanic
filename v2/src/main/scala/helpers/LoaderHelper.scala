package helpers

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame


object LoaderHelper {
  def loadFeatures(file:String):DataFrame = {
    // Spark import
    val ss = SparkHelper.getSession
    import ss.implicits._
    // Available columns: "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
    CSVHelper
      .load(file)
      .select("PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked")
      .map{row =>
        val id            = row.getAs[Int]("PassengerId").toDouble
        val pclass        = row.getAs[Int]("Pclass").toDouble
        val sex           = if (row.getAs[String]("Sex") == "male") 0D else 1D
        val age           = row.getAs[Double]("Age")
        val sibsp         = row.getAs[Int]("SibSp").toDouble
        val parch         = row.getAs[Int]("Parch").toDouble
        val fare          = row.getAs[Double]("Fare")
        val cabin         = row.getAs[String]("Cabin")
        val numberOfCabin = if (cabin != null) cabin.split(" ").length.toDouble else 0D
        val embarked      = row.getAs[String]("Embarked")
        val embarkedClass = if (embarked == "Q") 0D else if (embarked == "S") 1D else 2D
        (id, Vectors.dense(pclass, sex, age, sibsp/*, parch, fare*/, numberOfCabin, embarkedClass))
      }
      .withColumnRenamed("_1", "id")
      .withColumnRenamed("_2", "features")
  }

  def loadLabel(file:String):DataFrame = {
    CSVHelper
      .load(file)
      .select("PassengerId", "Survived")
      .withColumnRenamed("PassengerId", "id")
      .withColumnRenamed("Survived", "label")
  }
}

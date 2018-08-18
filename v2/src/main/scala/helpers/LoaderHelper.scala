package helpers

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame

// 0.7896678966789668 (pclass, sex)
// 0.8118081180811808 (pclass, sex, age) maxDeth 2 maxBIns 2
// 0.8118081180811808 (pclass, sex, age, fare) maxDeth 2 maxBIns 5
// 0.8154981549815498 (pclass, sex, age) maxDeth 5 maxBIns 10

// 0.7859778597785978 (pclass, sex, age, embarkedClass, numberOfCabin) maxDeth 5 maxBIns 30
// 0.8191881918819188 (pclass, sex, age, embarkedClass, sibsp) maxDeth 2 maxBIns 10
// 0.8376383763837638 (pclass, sex, age, embarkedClass, parch) maxDeth 2 maxBIns 30
// 0.8044280442804428 (pclass, sex, age, embarkedClass, fare)) maxDeth 2 maxBIns 5

// 0.8413284132841329 (pclass, sex, age, embarkedClass) maxDeth 2 maxBIns 30
// 0.8487084870848709

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
        (id, Vectors.dense(pclass, sex, age/*, sibsp, parch, fare, numberOfCabin*/, embarkedClass))
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

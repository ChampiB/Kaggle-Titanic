package helpers

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

object LoaderHelper {
  def loadFeatures(file:String):DataFrame = {
    // Spark import
    val ss = SparkHelper.getSession
    import ss.implicits._
    // Available columns: "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
    val data = CSVHelper
      .load(file)
      .select("PassengerId", "Name", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked")
      .map{row =>
        val id            = row.getAs[Int]("PassengerId").toDouble
        val pclass        = row.getAs[Int]("Pclass").toDouble
        val sex           = if (row.getAs[String]("Sex") == "male") 0D else 1D
        val age           = row.getAs[Double]("Age")
        val sibsp         = row.getAs[Int]("SibSp").toDouble
        val parch         = row.getAs[Int]("Parch").toDouble
        val family        = sibsp + parch
        val isAlone       = if (family == 0.0) 1 else 0
        val fare          = row.getAs[Double]("Fare")
        val cabin         = row.getAs[String]("Cabin")
        val numberOfCabin = if (cabin != null) cabin.split(" ").length.toDouble else 0D
        val embarked      = row.getAs[String]("Embarked")
        val embarkedClass = if (embarked == "Q") 0D else if (embarked == "S") 1D else 2D
        val title         = getTitle(row.getAs[String]("Name"))
        (id, pclass, sex, age, sibsp, parch, fare, cabin, numberOfCabin, embarked, embarkedClass, family, isAlone, title)
      }
      .withColumnRenamed("_1",  "id")
      .withColumnRenamed("_2",  "pclass")
      .withColumnRenamed("_3",  "sex")
      .withColumnRenamed("_4",  "age")
      .withColumnRenamed("_5",  "sibsp")
      .withColumnRenamed("_6",  "parch")
      .withColumnRenamed("_7",  "fare")
      .withColumnRenamed("_8",  "cabin")
      .withColumnRenamed("_9",  "numberOfCabin")
      .withColumnRenamed("_10", "embarked")
      .withColumnRenamed("_11", "embarkedClass")
      .withColumnRenamed("_12", "family")
      .withColumnRenamed("_13", "isAlone")
      .withColumnRenamed("_14", "title")
    // Filling empty data
    val meanAge = data.select(avg("Age")).collect().head.getAs[Double]("avg(Age)")
    val fillAge = udf((age:Double) => if (age == 0) meanAge else age)
    data.withColumn("Age", fillAge(col("Age")))
  }

  def getTitle(name:String):Double = {
    val mapping = Map(
      "rare" -> 0.0D,
      "ms" -> 1.0D,
      "master" -> 2.0D,
      "mme" -> 3.0D,
      "mr" -> 4.0D
    )

    val title = Array("([^ ]+)[.]".r.findFirstMatchIn(name).map(_.group(1)).map(_.toLowerCase).getOrElse("rare"))
      .map(t => if (Array("mme", "mrs") contains t) "mme" else t)
      .map(t => if (Array("miss", "mlle", "ms") contains t) "ms" else t)
      .map(t => if (Array("dr", "rev", "lady", "jonkheer", "capt", "col", "countess", "don", "sir", "major") contains t) "rare" else t)
      .head

    mapping.getOrElse(title, 0.0D)
  }

  def loadLabel(file:String):DataFrame = {
    CSVHelper
      .load(file)
      .select("PassengerId", "Survived")
      .withColumnRenamed("PassengerId", "id")
      .withColumnRenamed("Survived", "label")
  }
}

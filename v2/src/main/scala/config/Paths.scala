package config

object Paths {
  val resourcesBasePath = "./src/main/resources"
  val testSet = s"$resourcesBasePath/test.csv"
  val trainSet = s"$resourcesBasePath/train.csv"
  val model = s"$resourcesBasePath/model"
  val result = s"$resourcesBasePath/prediction"
}

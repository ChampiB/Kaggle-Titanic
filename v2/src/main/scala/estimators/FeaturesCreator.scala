package estimators

import org.apache.spark.sql.DataFrame

object FeaturesCreator {
  def create(modelName:String, data:DataFrame):DataFrame = {
    ModelMapping.get(modelName).createFeatures(data)
  }
}

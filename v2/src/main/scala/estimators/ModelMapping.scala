package estimators

import estimators.builders.{GBT, LogisticClassifier, MLP, MultiModelsClassifier}

object ModelMapping {
  private val mapping = Map[String, Builder](
    "GBT" -> new GBT,
    "MLP" -> new MLP,
    "Logistic" -> new LogisticClassifier,
    "MultiModels" -> new MultiModelsClassifier
  )
  def get(key:String):Builder = mapping(key)
}

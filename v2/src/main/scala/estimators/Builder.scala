package estimators

import org.apache.spark.ml.PipelineStage
import org.apache.spark.sql.DataFrame

abstract class Builder {
  def createFeatures(data:DataFrame):DataFrame
  def build(last:Boolean):PipelineStage
}

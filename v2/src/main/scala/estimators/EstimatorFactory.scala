package estimators

import org.apache.spark.ml.PipelineStage

object EstimatorFactory {
  def build(modelName:String, last:Boolean):PipelineStage = {
    ModelMapping.get(modelName).build(last)
  }
}

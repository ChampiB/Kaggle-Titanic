package trainers

import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.DataFrame

abstract class Trainer {
  def fit(data:DataFrame):TrainValidationSplitModel
}

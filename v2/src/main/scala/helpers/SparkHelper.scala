package helpers

import config.Application
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, SparkSession}

object SparkHelper {

  /*
  ** Spark session.
  */
  private var ss:SparkSession = _

  /*
  ** This function builds SparkSession.
  */
  def startSession(appName:String = Application.name): Unit = {

    if (ss != null) { this.stopSession() }
    ss = SparkSession
      .builder()
      .appName(appName)
      .master(Application.masterUrl)
      .getOrCreate()
  }

  /*
  ** This function returns the SparkSession.
  */
  def getSession: SparkSession = {

    if (ss == null) { this.startSession() }
    ss
  }

  /*
  ** This function returns the SparkContext.
  */
  def getSparkContext: SparkContext = {

    if (ss == null) { this.startSession() }
    ss.sparkContext
  }

  /*
  ** This function returns the SqlContext.
  */
  def getSqlContext: SQLContext = {

    if (ss == null) { this.startSession() }
    ss.sqlContext
  }

  /*
  ** This function delete SparkSession.
  */
  def stopSession(): Unit  = {

    if (ss != null) { ss.stop(); ss = null }
  }
}
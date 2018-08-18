package helpers

import org.apache.log4j.Logger
import org.apache.log4j.Level

object LoggerHelper {

  /*
  ** Turn off logging
  */
  def off():Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
  }
}

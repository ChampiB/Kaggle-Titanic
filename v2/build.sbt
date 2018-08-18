name := "v2"

version := "0.1"

scalaVersion := "2.11.8"

resolvers ++= Seq(
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Typesafe repository" at "https://repo.typesafe.com/typesafe/releases/"
)

val sparkVersion = "2.1.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)

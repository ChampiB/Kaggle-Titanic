name := "v2"

version := "0.1"

scalaVersion := "2.11.8"

// Dependencies
val sparkVersion = "2.1.0"
val dl4jVersion = "0.9.1"

libraryDependencies ++= Seq(
  "org.apache.spark"   %% "spark-core"          % sparkVersion % "provided",
  "org.apache.spark"   %% "spark-mllib"         % sparkVersion,
  "org.deeplearning4j" %  "deeplearning4j-core" % dl4jVersion,
  "org.deeplearning4j" %%  "dl4j-spark-ml"      % s"${dl4jVersion}_spark_2"
)

// Resolvers
resolvers += Resolver.mavenLocal
resolvers ++= Seq(
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Typesafe repository" at "https://repo.typesafe.com/typesafe/releases/"
)


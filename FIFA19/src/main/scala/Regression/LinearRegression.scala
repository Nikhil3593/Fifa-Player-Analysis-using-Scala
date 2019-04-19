import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}


object LinearRegression extends App {

  import org.apache.log4j._
  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

  val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("src/main/scala/Resources/data.csv")
  data.show()

  //features and labels
  import org.apache.spark.ml.feature.VectorAssembler
  import org.apache.spark.ml.linalg.Vectors

  import spark.implicits._
  val df = (data.select(data("Overall").as("label"), $"ID", $"Age", $"Special",
            $"Weak Foot", $"Skill Moves", $"crossing", $"Finishing", $"HeadingAccuracy", $"ShortPassing",
            $"Volleys", $"Dribbling", $"Curve", $"FKAccuracy", $"LongPassing", $"BallControl", $"Acceleration", $"SprintSpeed", $"Agility",
            $"Reactions", $"Balance", $"ShotPower", $"Jumping", $"Stamina", $"Strength", $"LongShots", $"Aggression", $"Interceptions", $"Positioning", $"Vision",
            $"Penalties", $"Composure", $"Marking", $"StandingTackle", $"SlidingTackle", $"GKDiving", $"GKHandling", $"GKKicking",
            $"GKPositioning", $"GKReflexes"
  ))

  df.printSchema()

  val assembler = (new VectorAssembler().setInputCols(Array("ID", "Age", "Special",
    "Weak Foot", "Skill Moves", "crossing", "Finishing", "HeadingAccuracy", "ShortPassing",
    "Volleys", "Dribbling", "Curve", "FKAccuracy", "LongPassing", "BallControl", "Acceleration", "SprintSpeed", "Agility",
    "Reactions", "Balance", "ShotPower", "Jumping", "Stamina", "Strength", "LongShots", "Aggression", "Interceptions", "Positioning", "Vision",
    "Penalties", "Composure", "Marking", "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling", "GKKicking",
    "GKPositioning", "GKReflexes")).setOutputCol("features"))

  val output = assembler.setHandleInvalid("skip").transform(df).select($"label", $"features")
  //Training and Testing
  var Array(training, test) = output.select($"label", $"features").randomSplit(Array(0.7,0.3), seed=1234)

  //output.show()

  val lr = new LinearRegression()

  //Train Split
//  val trainValSplit = (new TrainValidationSplit()
//      .setEstimator(lr)
//      .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
//      .setEstimatorParamMaps(paramGrid)
//      .setTrainRatio(0.8)
//    )
  val model = lr.fit(training)
  val testOutput = model.transform(test).select("features", "label", "prediction").show()
  val testSummary = model.summary
  println(testSummary.rootMeanSquaredError)
  println(testSummary.r2)


//  val lrModel = lr.fit(output)
//  println(s"coeff: ${lrModel.coefficients}, Intercept: ${lrModel.intercept}")
//
//  val trainingSummary = lrModel.summary
//  trainingSummary.predictions.show()
//  println("RMSE " + trainingSummary.rootMeanSquaredError)
//  println("R2 " + trainingSummary.r2)

}

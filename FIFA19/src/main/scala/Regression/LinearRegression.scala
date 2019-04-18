import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression



object LinearRegression extends App {

  import org.apache.log4j._
  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

  val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("src/main/scala/Resources/data.csv")

  //features and labels
  import org.apache.spark.ml.feature.VectorAssembler
  import org.apache.spark.ml.linalg.Vectors

  import spark.implicits._
  val df = (data.select(data("Potential").as("label"), $"ID", $"Age", $"Overall", $"Special",
            $"Weak Foot", $"Skill Moves", $"crossing", $"Finishing", $"HeadingAccuracy", $"ShortPassing",
            $"Volleys", $"Dribbling", $"Curve", $"FKAccuracy", $"LongPassing", $"BallControl", $"Acceleration", $"SprintSpeed", $"Agility",
            $"Reactions", $"Balance", $"ShotPower", $"Jumping", $"Stamina", $"Strength", $"LongShots", $"Aggression", $"Interceptions", $"Positioning", $"Vision",
            $"Penalties", $"Composure", $"Marking", $"StandingTackle", $"SlidingTackle", $"GKDiving", $"GKHandling", $"GKKicking",
            $"GKPositioning", $"GKReflexes"
  ))

  df.printSchema()

  val assembler = (new VectorAssembler().setInputCols(Array("ID", "Age", "Overall", "Special",
    "Weak Foot", "Skill Moves", "crossing", "Finishing", "HeadingAccuracy", "ShortPassing",
    "Volleys", "Dribbling", "Curve", "FKAccuracy", "LongPassing", "BallControl", "Acceleration", "SprintSpeed", "Agility",
    "Reactions", "Balance", "ShotPower", "Jumping", "Stamina", "Strength", "LongShots", "Aggression", "Interceptions", "Positioning", "Vision",
    "Penalties", "Composure", "Marking", "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling", "GKKicking",
    "GKPositioning", "GKReflexes")).setOutputCol("features"))

  val output = assembler.setHandleInvalid("skip").transform(df).select($"label", $"features")
  output.show()

  val lr = new LinearRegression()
  val lrModel = lr.fit(output)
  val trainingSummary = lrModel.summary
  trainingSummary.predictions.show()
  println(trainingSummary.r2)

}

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.regression.LinearRegressionModel



object LinearRegression {

  import org.apache.log4j._
  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

  val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("src/main/scala/Resources/data.csv")

  data.show()

}

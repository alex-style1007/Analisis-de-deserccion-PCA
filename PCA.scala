import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StandardScaler, PCA}
import org.apache.spark.ml.linalg.Vectors
import breeze.linalg._
import breeze.plot._

object DataProcessing {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Data Processing").getOrCreate()
    import spark.implicits._

    // Leer datos
    val data = spark.read
      .option("header", "true")
      .option("sep", ";")
      .option("encoding", "latin1")
      .option("inferSchema", "true")
      .csv("https://dominio/carpeta/permanencia_ingenierias.csv")

    data.describe().show()
    data.select("ALERTASMANUALES").show()

    data.head(5).foreach(println)

    data.groupBy("SEXO").count().show()

    // Eliminar columnas
    val filteredData = data.drop("ALERTASACADEMICAS", "ALERTASMANUALES")

    // Reemplazar valores
    val des = filteredData.withColumn("DESERSIÓN", when(col("DESERSIÓN") === "NO", 0).otherwise(1))

    val dataDummies = des.withColumn("SEXO", when(col("SEXO") === "M", 0)
      .when(col("SEXO") === "F", 1)
      .otherwise(2))
      .withColumn("APOYOS ECONÓMICO EN MATRICULA", when(col("APOYOS ECONÓMICO EN MATRICULA") === "NO", 0).otherwise(1))
      .withColumn("APOYO INSTITUCIONAL", when(col("APOYO INSTITUCIONAL") === "NO", 0).otherwise(1))
      .withColumn("ULTIMO ESTADO ACADÉMICO", when(col("ULTIMO ESTADO ACADÉMICO") === "Aprobado", 1).otherwise(0))

    // Normalizar datos
    val assembler = new org.apache.spark.ml.feature.VectorAssembler()
      .setInputCols(dataDummies.columns.filter(_ != "DESERSIÓN"))
      .setOutputCol("features")

    val featureData = assembler.transform(dataDummies)

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .fit(featureData)

    val scaledData = scaler.transform(featureData)

    // Calcular PCA
    val pca = new PCA()
      .setInputCol("scaledFeatures")
      .setOutputCol("pcaFeatures")
      .setK(2)
      .fit(scaledData)

    val pcaResult = pca.transform(scaledData).select("pcaFeatures")

    // Convertir a formato Breeze para visualización
    val pcaFeatures = pcaResult.collect().map(_.getAs[org.apache.spark.ml.linalg.Vector](0).toArray)
    val breezeMatrix = DenseMatrix(pcaFeatures: _*)

    // Graficar resultados
    val f = Figure()
    val p = f.subplot(0)
    p += plot(breezeMatrix(::, 0), breezeMatrix(::, 1), '+', colorcode = "blue")
    p.xlabel = "PC1"
    p.ylabel = "PC2"
    p.title = "PCA Result"

    // Mostrar gráficos
    f.refresh()

    spark.stop()
  }
}

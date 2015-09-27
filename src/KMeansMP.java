import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;


public final class KMeansMP {
	private static class ParsePoint implements Function<String, Vector> {
		private static final Pattern SPACE = Pattern.compile(",");

		public Vector call(final String line) {
			final String[] tok = SPACE.split(line);
			final double[] point = new double[tok.length-1];

			for (int i = 1; i < tok.length; ++i) {
				point[i-1] = Double.parseDouble(tok[i]);
			}

			return Vectors.dense(point);
		}
	}

	private static class ParseTitle implements Function<String, String> {
		private static final Pattern SPACE = Pattern.compile(",");

		public String call(final String line) {
			final String[] tok = SPACE.split(line);

			return tok[0];
		}
	}

	private static class ClusterCars implements PairFunction<Tuple2<String, Vector>, Integer, String> {
		private KMeansModel model;

		public ClusterCars(final KMeansModel model) {
			this.model = model;
		}

		public Tuple2<Integer, String> call(final Tuple2<String, Vector> args) {
			final String title = args._1();
			final Vector point = args._2();
			final int cluster = model.predict(point);

			return new Tuple2<Integer, String>(cluster, title);
		}
	}

	public static void main(String[] args) {
		if (args.length < 2) {
			System.err.println("Usage: KMeansMP <input_file> <results>");

			System.exit(1);
		}

		String inputFile = args[0];
		String results_path = args[1];

		int k = 4;
		int iterations = 100;
		int runs = 1;
		long seed = 0;
		
		SparkConf sparkConf = new SparkConf().setAppName("KMeans MP");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		JavaRDD<String> lines = sc.textFile(inputFile);
		JavaRDD<Vector> points = lines.map(new ParsePoint());
		JavaRDD<String> titles = lines.map(new ParseTitle());

		final KMeansModel model = KMeans.train(points.rdd(),
						       k,
						       iterations,
						       runs,
						       KMeans.RANDOM(),
						       seed);

		final JavaPairRDD<Integer, Iterable<String>> results = titles.zip(points)
									     .mapToPair(new ClusterCars(model))
									     .groupByKey();
		
		results.saveAsTextFile(results_path);

		sc.stop();
	}
}


import java.io.*;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;


public class Jacobi extends Configured {

	public interface Debug   
	{    
		public final boolean ENABLE = true;    
	}  

	static enum Counters { EPS_COUNTER }

	public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, DoubleWritable> {

		private double[] x ;
		private int iteration ;
		private int size;

		public void configure(JobConf job) {

			size = job.getInt("size", 0);
			iteration = job.getInt("iteration", job.getInt("max_iter",100));

			if (size==0||iteration==100) {System.out.println("Matrix size/iteration error detected");System.exit(0);}
			this.x = new double[size];

			for(int i=0;i<size;i++)
				x[i] = 0;
			if(Debug.ENABLE)
			{
				System.out.println("[Map Config] size:"+size+" iteration:"+iteration);
			}
			FSDataInputStream    fs ;
			FileSystem hdfs ;
			URI[] vectorFiles;
			Path mypath ;
			if (iteration==0) return;
			try {
				hdfs = FileSystem.get(job);
				vectorFiles = DistributedCache.getCacheFiles(job);
				mypath = new Path(vectorFiles[1].getPath());
				fs = null;
				if(hdfs.exists(mypath))
				{
					fs=hdfs.open(mypath);
					if(Debug.ENABLE)
					{
						System.out.println("[Map Config] success open:"+mypath.toString() );
					}
				}
				else
					System.err.println("[Map Config] fail open:"+mypath.toString() );

				if(Debug.ENABLE)
				{
					System.out.println("[Map Config] vectorFiles size:"+vectorFiles.length );
					System.out.println("[Map Config] CacheFile Path:"+ job.get("mapred.cache.files") );
				}
				String element = null;

				int i=0;
				while ((element = fs.readLine()) != null && i<size) {
					x[i] = Double.parseDouble(element.trim().split("\\s+")[1]);
					i++;		
				}
				fs.close();
			} catch (IOException ioe) {
				System.err.println("Caught exception while processing the cached file '" + StringUtils.stringifyException(ioe));
			}

		}

		public void map(LongWritable key, Text value, OutputCollector<IntWritable,DoubleWritable> output, Reporter reporter) throws IOException {
			try{
				String[] tmp = value.toString().trim().split("\\s+");
				int row = Integer.parseInt(tmp[0]);
				int column = Integer.parseInt(tmp[1]);
				double val = Double.parseDouble(tmp[2]);
				if (row!=column&&column!=size)
					output.collect(new IntWritable(row), new DoubleWritable(val*x[column]));
			}catch(Exception e){System.err.println("[Map] Caught exception while a[i][j]*P[j] " + StringUtils.stringifyException(e));}
		}
	}


	public static class Combiner extends MapReduceBase implements Reducer<IntWritable,DoubleWritable, IntWritable,DoubleWritable> {

		public void reduce(IntWritable key, Iterator<DoubleWritable> values, OutputCollector<IntWritable,DoubleWritable> output, Reporter reporter) throws IOException {
			double sum = 0;
			while (values.hasNext()) {
				sum += values.next().get();
			}
			output.collect(key, new DoubleWritable(sum));
		}
	}

	public static class Reduce extends MapReduceBase implements Reducer<IntWritable,DoubleWritable, IntWritable,DoubleWritable> {

		private int iteration ;
		private int size;
		private double eps;
		private double[] diagonal;
		private double[] R;
		private double[] x ;

		public void configure(JobConf job) {

			size = job.getInt("size", 0);
			iteration = job.getInt("iteration", job.getInt("max_iter",100));
			eps = Double.valueOf(job.getStrings("eps", "1e-10")[0]);

			if (size==0||iteration==100) {System.out.println("Matrix size/iteration error detected");System.exit(0);}
			this.diagonal = new double[size];
			this.R = new double[size];
			this.x = new double[size];

			for(int i=0;i<size;i++)
			{
				R[i] = 0;
				diagonal[i] = 0;
				x[i] = 0;
			}
			if(Debug.ENABLE)
			{
				System.out.println("[Reduce Config] size:"+size+" iteration:"+iteration+" eps"+eps );
			}
			FSDataInputStream    fs ;
			FileSystem hdfs ;
			URI[] vectorFiles;
			Path mypath ;
			try {
				hdfs = FileSystem.get(job);
				vectorFiles = DistributedCache.getCacheFiles(job);

				/* read matrix file */
				mypath = new Path(vectorFiles[0].getPath());
				fs = null;
				if(hdfs.exists(mypath))
				{
					fs=hdfs.open(mypath);
					if(Debug.ENABLE)
					{
						System.out.println("[Reduce Config] success open:"+mypath.toString() );
					}
				}
				else
					System.err.println("[Reduce Config] fail open:"+mypath.toString() );

				if(Debug.ENABLE)
				{
					System.out.println("[Reduce Config] vectorFiles size:"+vectorFiles.length );
					System.out.println("[Reduce Config] CacheFile Path:"+ job.get("mapred.cache.files") );
				}
				String element = null;

				String[] tmp;
				int row;
				int column;
				double val;
				while ((element = fs.readLine()) != null) {
					tmp = element.trim().split("\\s+");
					row = Integer.parseInt(tmp[0]);
					column = Integer.parseInt(tmp[1]);
					val = Double.parseDouble(tmp[2]);

					if (row==column)
						diagonal[row] = val;
					else if (column==size)
						R[Integer.parseInt(tmp[0])]= Double.parseDouble(tmp[2]);
				}
				fs.close();

				/* read immediate result file from last iteration */
				if(iteration!=0)
				{
					mypath = new Path(vectorFiles[1].getPath());
					fs = null;
					if(hdfs.exists(mypath))
					{
						fs=hdfs.open(mypath);
						if(Debug.ENABLE)
						{
							System.out.println("[Reduce Config] success open:"+mypath.toString() );
						}
					}
					else
						System.err.println("[Reduce Config] fail open:"+mypath.toString() );

					if(Debug.ENABLE)
					{
						System.out.println("[Reduce Config] vectorFiles size:"+vectorFiles.length );
						System.out.println("[Reduce Config] CacheFile Path:"+ job.get("mapred.cache.files") );
					}
					element = null;

					int i=0;
					while ((element = fs.readLine()) != null && i<size) {
						x[i] = Double.parseDouble(element.trim().split("\\s+")[1]);
						i++;		
					}
					fs.close();
				}
			} catch (IOException ioe) {
				System.err.println("Caught exception while processing the cached file '" + StringUtils.stringifyException(ioe));
			}

		}
		public void reduce(IntWritable key, Iterator<DoubleWritable> values, OutputCollector<IntWritable,DoubleWritable> output, Reporter reporter) throws IOException {
			double sum = 0;
			while (values.hasNext()) {
				sum += values.next().get();
			}
			double new_x = (R[key.get()]-sum)/diagonal[key.get()];
			System.out.println("[Reduce] Index: "+ key.get()+" sum"+ sum + " R[key.get()]"+R[key.get()]+" diagonal[key.get()]"+diagonal[key.get()]);
			output.collect(key, new DoubleWritable(new_x));
			if (Math.abs(new_x-x[key.get()])>eps)
			{
				System.out.println("[Reduce] Counter ++ Index: "+ key.get());
				reporter.incrCounter(Counters.EPS_COUNTER, 1);
			}
		}
	}

	public static void main(String[] args) throws Exception {

		if (args.length<2||args.length>3)
		{
			System.out.println("Usage: <input matrix> <output dir> [(optional)number of map tasks]");
		}

		long counter = 1;
		int iteration = 0;
		int size = Integer.parseInt(args[0].substring(args[0].lastIndexOf('/')+1, args[0].indexOf(".dat")));
		double eps = 1e-10;
		int max_iter = 100;

		while (counter!=0 && iteration < max_iter )
		{

			JobConf conf = new JobConf(Jacobi.class);
			if(args.length==3)
			{
				conf.set("mapred.map.tasks", args[2]);		
				if(Debug.ENABLE)
				{
					System.out.println("[Config] Number of map: "+args[2]);
				}
			}
			conf.setJobName("jacobi"+"_"+iteration);
			conf.setInt("iteration", iteration);
			conf.setInt("size", size);
			conf.setStrings("eps", Double.toString(eps));
			conf.setInt("max_iter", max_iter);
			conf.setMapOutputKeyClass(IntWritable.class);
			conf.setMapOutputValueClass(DoubleWritable.class);
			conf.setOutputKeyClass(Text.class);
			conf.setOutputValueClass(IntWritable.class);

			conf.setMapperClass(Map.class);
			conf.setCombinerClass(Combiner.class);
			conf.setReducerClass(Reduce.class);
		    conf.setNumMapTasks(10);
			conf.setInputFormat(TextInputFormat.class);
			conf.setOutputFormat(TextOutputFormat.class);

			/* add matrix file to system*/
			Path des = new Path(args[0]);
			if(Debug.ENABLE)
			{
				System.out.println("[Conf] Iteration:"+ iteration +" Matrix File Path: "+ des.toString());
			}
			DistributedCache.addCacheFile(des.toUri(),conf);

			/* add immediate x result file to system*/
			if(iteration!=0)
			{
				Path desX = new Path(args[1]+"/"+(iteration-1)+"/part-00000");
				if(Debug.ENABLE)
				{
					System.out.println("[Conf] Iteration:"+ iteration +" Immediate Result File Path: "+ desX.toString());
				}
				DistributedCache.addCacheFile(desX.toUri(),conf);				
			}
			FileInputFormat.setInputPaths(conf, new Path(args[0]));
			FileOutputFormat.setOutputPath(conf, new Path(args[1]+"/"+iteration));
			RunningJob parentJob = JobClient.runJob(conf);
			counter = parentJob.getCounters().getCounter(Counters.EPS_COUNTER);
			if(Debug.ENABLE)
			{
				System.out.println("[Conf] Counter:"+counter);
			}
			iteration ++ ;
		}
		System.out.println("Jacobi use "+ iteration +" iterations to solve "+size+".dat");
		System.out.println("Output file: hdfs://node17.cs.rochester.edu:9000"+args[1]+"_"+(iteration-1)+"/part-00000");
		System.exit(0);

	}
}

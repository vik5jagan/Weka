package org.cluster;


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
 
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
 
public class KMeansAlgorithm {
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
 
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile("c:\\ads.txt");
 
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
 
		Classifier ibk = new IBk();		
		ibk.buildClassifier(data);
 
		BufferedReader testfile = readDataFile("c:\\data\\testdata.txt");
		 
		Instances testData = new Instances(testfile);
		testData.setClassIndex(testData.numAttributes() - 1);
		
		//do not use first and second
		Instance first = testData.instance(0);
		Instance second = testData.instance(1);
		  
		
		double class1 = ibk.classifyInstance(first);
		double class2 = ibk.classifyInstance(second);
 
		System.out.println("first: " + class1 + "\nsecond: " + class2);
	}
}
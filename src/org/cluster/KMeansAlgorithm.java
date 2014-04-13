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
		BufferedReader datafile = readDataFile("C:\\Users\\SONY\\git\\Weka\\src\\org\\cluster\\ads.txt");
 
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
 
		Classifier ibk = new IBk();		
		ibk.buildClassifier(data);
 
		BufferedReader testfile = readDataFile("C:\\Users\\SONY\\git\\Weka\\src\\org\\cluster\\testdata.txt");
		 
		Instances testData = new Instances(testfile);
		testData.setClassIndex(testData.numAttributes() - 1);
		
		//do not use first and second
		Instance first = testData.instance(0);
		Instance second = testData.instance(1);
		Instance third = testData.instance(2);
		Instance four = testData.instance(3);
		
		double class1 = ibk.classifyInstance(first);
		double class2 = ibk.classifyInstance(second);
		double class3 = ibk.classifyInstance(third);
		double class4 = ibk.classifyInstance(four);
		
		System.out.println("first: " + class1 + "\nsecond: " + class2 + "\nthird: " + class3 + "\nfour: " + class4);
	}
}
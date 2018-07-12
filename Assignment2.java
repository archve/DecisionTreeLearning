import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.TreeMap;

/*class Assignment has methods for the three tasks of assignemnt2
 * Task 1 - decision tree through ID3
 * Task 2 - Redued Error Pruning
 * Task 3 - Implementing random forest
 * */
class treeNode { // Node structure of the decision tree
			int nodeType; // can be 0 or 1: 0 -> intermediate node , 1->leaf node holding target functionval in label field
			int index; // index of this feature in the feature list
			int c_flag; // =1 implies the node attribue is continuous type
			int splitValue; // splitval of continuous attribute
			String nodeLabel;
			String domainValue; // domain value of parent thorugh which this node was reached
			ArrayList<treeNode> children = new ArrayList<treeNode>();
}

class Accuracy {
	//This class holds the performance statistics of the algorithm	
	int actualTargetVal1 = 0,actualTargetVal2 = 0,trueTarget1 = 0,falseTarget1 = 0,trueTarget2 = 0;
	int falseTarget2 = 0;
	float accuracy;
	float t1precision;
	float t1recall;
	float t1Fmeasure;
	float t2precision;
	float t2recall;
	float t2Fmeasure;
	/* the following methods return the specific performance measures for the target function value passed
	 * tarVal = 0 ---> >50K  i.e targetfunction value 1
	 * tarVal = 1 ---> <=50K i.e targetfunction value 2
	 * */
	float calculatePrecision(int targetVal){
		float precision = 0;
		float temp=0;
		if(targetVal == 0) {
			temp = trueTarget1+falseTarget1;
			t1precision = (float)trueTarget1/temp;
			precision = t1precision;
		}else {
			temp = trueTarget2+falseTarget2;
			t2precision = (float)trueTarget2/temp;
			precision = t2precision;
		}
		return precision;		
	}
	
	float calculateRecall(int targetVal){
		float recall = 0;
		float temp=0;
		if(targetVal == 0) {			
			temp =trueTarget1+falseTarget2;
			t1recall = (float)trueTarget1/temp;
			recall =t1recall;
		}else {
			temp =trueTarget2+falseTarget1;
			t2recall = (float)trueTarget2/temp;
			recall =t2recall;
		}
		
		return recall;
		
	}
	float calculatefMeasure(int targetVal){
		float fMeasure= 0;
		float p,r;
		p = calculatePrecision(targetVal);
		r = calculateRecall(targetVal);
		if(targetVal == 0) {
			t1Fmeasure = (2 * r * p) /(r+p);
			fMeasure = t1Fmeasure;
			
		}else {
			t2Fmeasure = (2 * r * p) /(r+p);
			fMeasure = t2Fmeasure;
		}
		
		return fMeasure;		
	}

}
public class Assignment2 {
	static ArrayList<String> targetFuntionValues= new ArrayList<String>();
	static int targetcount;
	static int FeaturesCount=0;
	static int target1,target2;
	static int continuousFeatureCount=0;
	static double globalEntropy =0; // not using the global value anywhere. remove and make local?
	static String featureFile= "dataset/attributedetails.txt";
	static String trainingDataFilename = "dataset/adultdata.txt";
	static String testingDataFilename = "dataset/adulttest.txt";
	static ArrayList<String> maxDomainvalue ;
	static ArrayList<ArrayList<String>> trainData2;
	static ArrayList<ArrayList<String>> validatingData;
	static ArrayList<String> featureNames = new ArrayList<String>();	
	static ArrayList<Integer> continuousFeatureIndex =new ArrayList<Integer>(); // indices of feature from feature names which are continuous
	static ArrayList<ArrayList<String>> testingData = new ArrayList<ArrayList<String>> (); // each value has a list of attributes
	static ArrayList<ArrayList<String>> trainingData = new ArrayList<ArrayList<String>> (); // each value has a list of attributes
	static ArrayList<ArrayList<String>> featureDomain = new ArrayList<ArrayList<String>>(); // domain values at index i are of a feature in featureNames at index i	
	static HashMap<Integer,Integer> continuousValSplits = new HashMap<Integer,Integer>();
	static ArrayList<ArrayList<Integer>> valueCount = new ArrayList<ArrayList<Integer>>();
	

		
	static treeNode train(ArrayList<ArrayList<String>> data, ArrayList<String> tarVal, ArrayList<Integer> aSet, int algoType, ArrayList<Integer> remainingAttributes, HashMap<Integer,Integer>continuousValSplits) {
			/* recursively builds the decision tree
			 * Returns the reference to the tree constructed on the data set and feature set passed
			 * works for both ID3 and Random forest
			 * algoType 0-> ID3/pruning
			 * 			1-> RandomForest
			 * */
			treeNode node = null;
			//variables to calculate entropy and store the max target funtion val for the node
	   		int maxIndex = 0; 
			int totalCount = 0;
			int maxTarIndex ;//index val of max target attribute
			double max = 0;
			double dEntropy =0;
			double [] tarCount =  new double[tarVal.size()];
			double[] wEntropy = new double[aSet.size()];//weighted entropy in each case
			ArrayList<Double> infoGain = new ArrayList<Double>(aSet.size()); 
	  
			// find count of each target attribute instance in data and calculate entropy
   		    for(int dIndex =0 ; dIndex < data.size();dIndex++) {
   		    	String  target=data.get(dIndex).get(FeaturesCount);
						if(target.equals(tarVal.get(0))) {
							tarCount[0] =tarCount[0]+1;
   							totalCount = totalCount +1;

						}
						else {
							tarCount[1] =tarCount[1]+1;
   							totalCount = totalCount +1;
						}						
   		    }
   		    dEntropy = calculateEntropy(tarCount[0],tarCount[1],totalCount);
   		    // if entropy is zero then node is leaf type , return leaf node
   		    if(tarCount[0]==0 || tarCount[1]==0) {
   		    	node = new treeNode();
   		    	node.nodeType = 1;
   		    	if(tarCount[0] != 0) {
   		    		node.nodeLabel = targetFuntionValues.get(0); 
   		    	}
   		    	else { // assuming both won't be zero at the same time.
   		    		node.nodeLabel = targetFuntionValues.get(1); 
   		    	}
   		    	return node;
   		    }
   		    
   		    // for each feature in aSet calculating information gain
   		    for(int index =0 ; index < aSet.size();index++) {
   		    	int featureIndex = aSet.get(index);
   		    	int c_flag = 0; // continuous attribute flag
   		    	ArrayList<String> domain= featureDomain.get(featureIndex);
   		    	int domainCount = domain.size();

   		    	if(domainCount == 1 && domain.contains("continuous")) {
   		    		   		c_flag = 1;
   		    	}
   		    	
   		    	
   		    	if(c_flag ==1) { // entropy of continuous feature with binary split
	    		   	int splitval = continuousValSplits.get(featureIndex);

   		    	//local variables for entropy calculation for less than split value
   					double lt_total=0;
   					double lt_target1=0; 
   					double lt_target2=0;
   					double lt_entropy=0;
   					//local variables for entropy calculation for less than split value
   					double gt_total=0;
   					double gt_target1=0; 
   					double gt_target2=0;
   					double gt_entropy=0;
   					double l_weightedEntropy=0;
   					double l_infoGain = 0;
   					
   					// Calculating entropy
   					for(int dIndex =0 ; dIndex < data.size();dIndex++) {
						String value =data.get(dIndex).get(FeaturesCount);
   						if(Integer.parseInt(data.get(dIndex).get(featureIndex)) <= splitval) {
   							
   							lt_total = lt_total +1;
   							if(value.equals(tarVal.get(0)))
   									lt_target1 = lt_target1+1;
   							else
   									lt_target2 = lt_target2+1;


   						}
   						else {
   							gt_total = gt_total +1;
   							if(value.equals(tarVal.get(0)))
   									gt_target1 = gt_target1+1;
   							else
   								gt_target2 = gt_target2+1;  							
   						}
   						
   						
   					}	
   					lt_entropy = calculateEntropy(lt_target1,lt_target2,lt_total);
   					gt_entropy = calculateEntropy(gt_target1,gt_target2,gt_total);
   					if((lt_total+gt_total) >0) {
   						l_weightedEntropy = (lt_total * lt_entropy + gt_total * gt_entropy) / (lt_total + gt_total);
   					}
   					wEntropy[index] = l_weightedEntropy;
   		    		l_infoGain = dEntropy - l_weightedEntropy;
   		    		infoGain.add(index, l_infoGain);   		    		
   		    	}   		        
   		    	else { // entropy of features with discreate values
   		    		double l_weightedEntropy = 0;
   		    		double dTotal =0 ;
   					double l_infoGain = 0;
   		    		double[] l_total= new double[domainCount];
   	   				double[] l_target1=new double[domainCount]; 
   	   				double[] l_target2=new double[domainCount];
   	   				double[] l_entropy=new double[domainCount]; 
   	   				for(int dIndex =0 ; dIndex < data.size();dIndex++) {
   	   					String val = data.get(dIndex).get(featureIndex);
   	   					if(val.contains("?")) {
   	   						val = maxDomainvalue.get(featureIndex);
   	   						data.get(dIndex).set(featureIndex, val);
   	   					}
   	   					int dom_index = domain.indexOf(val);
   	   					if(dom_index >= 0) { // skipping missing values for now   	   							
   	 	   					l_total[dom_index]=l_total[dom_index]+1;
   	   						String  target=data.get(dIndex).get(FeaturesCount);
   	   						if(target.equals(tarVal.get(0)))
   	   							l_target1[dom_index]=l_target1[dom_index]+1;
   	   						else
   	   							l_target2[dom_index]=l_target2[dom_index]+1;
   	   					} 	   						
   	    		    }
   	   				for(int dom =0 ; dom < domainCount;dom++) {
   	   					l_entropy[dom] =  calculateEntropy(l_target1[dom],l_target2[dom],l_total[dom]);
   	   					l_weightedEntropy =  l_weightedEntropy + (l_total[dom] * l_entropy[dom]);
   	   					dTotal = dTotal +l_total[dom] ;
	    		    } 
   	   				if(dTotal > 0 && l_weightedEntropy > 0) {
   	   					l_weightedEntropy =l_weightedEntropy / dTotal;
   	   				}
   	   			wEntropy[index] = l_weightedEntropy;
		    	l_infoGain = dEntropy - l_weightedEntropy;
   		    	infoGain.add(index, l_infoGain);
   		    }    	
   		   } 		    
   		    
   		    max = Collections.max(infoGain); // max val of informatin gain amongst all 
   		    maxIndex = infoGain.indexOf(max); // index with max infoGain
   		    
   		 	int valIndex = aSet.get(maxIndex); // actual index of the feature selected with max gain
   		 	// node attributes
   		    node = new treeNode();
   		 	node.index = valIndex;
   		    node.nodeLabel =featureNames.get(valIndex);
		   // get domain values of the selected feature
   		    ArrayList<String> domain= featureDomain.get(valIndex);
		    int domainCount = domain.size();
		    int c_flag = 0; 
	    	if(domainCount == 1 && domain.contains("continuous")) {
	    		c_flag = 1;
	    		node.c_flag = 1;	    		
	    		domainCount = 2;
	    	}
   		    //creating array for attribute and removing the selected index from it aSet
    		ArrayList<Integer> newaSet = new ArrayList<Integer>();
    		ArrayList<Integer> newRA = new ArrayList<Integer>();
	    	if(algoType == 0) {
	    		for (int i = 0 ; i<aSet.size();i++){	    			
	    			newaSet.add(aSet.get(i)) ;
	    		}	    	
	    		newaSet.remove(maxIndex); // remove the selected feature from the set,this is passed calculate the next node
	    		remainingAttributes = newaSet; 
	    	}else {
	    		for (int i = 0 ; i<remainingAttributes.size();i++){	    			
	    			newRA.add(remainingAttributes.get(i)) ;
	    		}
	    		int indexToDelete = newRA.indexOf(valIndex);
	    		newRA.remove(indexToDelete);
	    	}
   			//flitering data based in the attribute value,to pass to the next value 
   		    for(int dom =0 ; dom < domainCount;dom++) { // for value in the domain of that particular domain  
   		    	if(algoType == 1) {   		    		
   					newaSet = getFeatureSubset(remainingAttributes);	    		
   		    	}
   		    	
   		    	ArrayList<ArrayList<String>> domdata = new ArrayList<ArrayList<String>> ();										  
   		    	for (int i = 0; i < data.size(); i++)
   		    	{	ArrayList<String> datapt = data.get(i);
   		    		if(c_flag == 1) {
   	   		    		int val =Integer.parseInt(datapt.get(valIndex));
   	   		    		int splitval = continuousValSplits.get(valIndex);

   	   		    		if(dom == 0) {
   	   		    			if(val <= splitval) {
   	   		    				domdata.add(datapt);
   	   		    			}
   	   		    		}   	   		    			
   	   		    		else if (dom==1){
   	   		    			
   	   		    			if(val > splitval) {
	   		    				domdata.add(datapt);
	   		    			}
   	   		    		}

   		    		}
   		    		else {
   		    			String val =datapt.get(valIndex);
   		   		    	String domValue = domain.get(dom);
   		    			if( val.equals(domValue)){
   		    				domdata.add(datapt);
   		    			}
   		    		}
   		              
   		    	}
   		    	
   		    	HashMap<Integer,Integer> newcontinuousSplitvalue = discretizeData(domdata); // calculating split values for continuous attributes
   		    	if(domdata.size() == 0 || (domdata.size() == data.size())|| newaSet.isEmpty()||remainingAttributes.isEmpty()) { // leaf nodes of the tree
   		    		treeNode leaf = new treeNode(); 
   		    		leaf.nodeType =1;
   		    		if(tarCount[0] > tarCount[1]) {
   		    			maxTarIndex = 0;
   		    		}else {
   		    			maxTarIndex = 1;
   		    		}
		    		leaf.nodeLabel = targetFuntionValues.get(maxTarIndex);
		    		if(c_flag == 1) {
	   		    		int splitval = continuousValSplits.get(valIndex);

	   		    		if(dom == 0) {
	   		    			node.splitValue = splitval;
	   	   		    		leaf.domainValue = "<"+splitval;

	   		    		}else if(dom == 1){
	   		    			node.splitValue = splitval;
	   	   		    		leaf.domainValue = ">"+splitval;
	   		    		}   		    		
		    		}
		    		else {
		   		    	String domValue = domain.get(dom);
		    			leaf.domainValue = domValue;
		    		}

   		    	    node.children.add(leaf);
   		    	}   		    	
   		    	else { // intermediate nodes 
   	   		    	treeNode child = null;
   	   		    	child = train(domdata,tarVal,newaSet,algoType,remainingAttributes,newcontinuousSplitvalue); 	  // next node
   	   		    	if(c_flag == 1) {
   	   		    		int splitval = continuousValSplits.get(valIndex);

   	   		    		if(dom == 0) {
	   		    			node.splitValue = splitval;
   	   		    			child.domainValue = "<"+splitval;

   	   		    		}else if(dom == 1){
	   		    			node.splitValue = splitval;
   	   		    			child.domainValue = ">"+splitval;
   	   		    		}		    		
   	   		    	}
   	   		    	else {
   	    		    	String domValue = domain.get(dom);
   	   		    		child.domainValue = domValue;
   	   		    	}
   	   		    	node.children.add(child);
   		    	} 	
   		    } 
			return node;
		}
	
	static Accuracy test(treeNode root, ArrayList<ArrayList<String>> data) {
		
		//int actualTargetVal1 = 0,actualTargetVal2 = 0,trueTarget1 = 0,falseTarget1 = 0,trueTarget2 = 0,falseTarget2 = 0;
		//System.out.println("Data size is: "+data.size());
		Accuracy ob=new Accuracy();
		for(int dIndex=0;dIndex < data.size();dIndex++ ) {
			ArrayList<String> datapt = new ArrayList<String>();
			datapt = data.get(dIndex);
			String result = getResult(root,datapt);
			if(result != null) {
				String actualResult = datapt.get(FeaturesCount);
				if (actualResult.endsWith(".")) {
					actualResult = actualResult.substring(0, actualResult.length() - 1);
				}
				if(actualResult.equals(targetFuntionValues.get(0))) {
					ob.actualTargetVal1 = ob.actualTargetVal1 +1;
					if(result.equals(actualResult)) {
						ob.trueTarget1 = ob.trueTarget1 +1;
					}else {
						ob.falseTarget2 = ob.falseTarget2 + 1;
					}				
				}else if(actualResult.equals(targetFuntionValues.get(1))) {
					ob.actualTargetVal2 = ob.actualTargetVal2 + 1;
					if(result.equals(actualResult)) {
						ob.trueTarget2 = ob.trueTarget2 +1;
					}else {
						ob.falseTarget1 = ob.falseTarget1 + 1;
					}				
				}
			
		   }		
		}
		int temp=(ob.trueTarget1+ob.trueTarget2+ob.falseTarget1+ob.falseTarget2);
		if(temp==0) ob.accuracy=0;
		else
		ob.accuracy=(float)(ob.trueTarget1+ob.trueTarget2)/temp;		
		return ob;
		//System.out.println("actualTargetVal1 ="+actualTargetVal1+" actualTargetVal2="+actualTargetVal2+" trueTarget1 = "+trueTarget1+" falseTarget1 = "+falseTarget1+"	trueTarget2 ="+trueTarget2+" falseTarget2 = "+falseTarget2);
}
	
	private static String getResult(treeNode node, ArrayList<String> instance) {
		// recursively traverses down the path on the decision tree and returns the result		
		if(node.nodeType == 1) { // if its a leaf node , we have got the result
			return node.nodeLabel;
		}
		else { // intermediate node , need to jump to next node for result
			int featureListIndex = node.index; // gives us index in instance to look for the value for comparison
			String featureValue = instance.get(featureListIndex);
			String lookupVal;
			if(featureValue.contains("?")) { //for data with missing attribute values
				featureValue = maxDomainvalue.get(featureListIndex);
			}
			treeNode nextNode = null;   
			if(node.c_flag==1) {
				int split = node.splitValue;
				int val =Integer.parseInt(featureValue);
				if(val <= split) {
					lookupVal = "<"+split;
				}	
				else {
					lookupVal = ">"+split;
				}
			}
			else {
				lookupVal = featureValue; 
			}
			
		for(int i = 0 ; i < node.children.size();i++) {
			if(lookupVal.equals(node.children.get(i).domainValue)) {
				nextNode = node.children.get(i);
				break;
			}
		}
		if(nextNode == null) return null; // not possible but just in case		
		return getResult(nextNode,instance);
	}
}

	static double calculateEntropy(double a,double b,double t){
			double val1 = 0 ;
			double val2 = 0 ;
			double temp1;
			double temp2;
			double result;
			if((t!= 0 )&& (a!=0)) {
				val1 = a/t;
			}
			if((t!= 0 )&& (b != 0)) {
				val2 = b/t;
			}
			if(val1 == 0) {
				temp1 = 0;
			}else {
				double log_temp1 = Math.log10(val1);
				double log_temp2 = Math.log10(2);
				temp1= log_temp1/log_temp2;
				//temp1 = Math.log10(val1);

				
			}
			
			if(val2 == 0) {
				temp2 = 0;
			}else {
				double log_temp1 = Math.log10(val2);
				double log_temp2 = Math.log10(2);
				temp2= log_temp1/log_temp2;
				
				//temp2 = Math.log10(val2);
			}
			result = (val2*temp2) + (val1*temp1);
			if(result == 0) {
				return 0;
			}
			else {
				return (-result);
			}
		}
	
	
	public static void loadFeatures(String inpFile){
		/* This method 
		 * 			-reads the features names of the dataset from the file into <featureNames> 
		 * 			- maintains count for features
		 * 			-reads the possible domain values of each features into <featureDomain>
		 * 			-gets the target function values into <targetFunctionValues>
		 * 			- for continuous features, creates a list of their indices
		 *  */	
		try{
			BufferedReader br = new BufferedReader(new FileReader(inpFile));
			String line;
			line = br.readLine();
			StringTokenizer tokens = new StringTokenizer(line,",");
			targetcount = tokens.countTokens();
			
			// first line holds the details of the target funtion 
			while (tokens.hasMoreTokens()) {
				String tokenVal = tokens.nextToken().trim();
				if (tokenVal.endsWith(".")) {
					tokenVal = tokenVal.substring(0, tokenVal.length() - 1);
				}
				targetFuntionValues.add(tokenVal);
		     }	
			while ((line = br.readLine()) != null) {
				if(line.length() > 0) { // to skip the blank line after target function values
					StringTokenizer AttributeString = new StringTokenizer(line,":"); //separating attribute label and its domain values
					String AttributeName;
					String AttributeDomain = null;
					
					//feature names
					if(AttributeString.hasMoreTokens()) {
						AttributeName=AttributeString.nextToken();
						featureNames.add(AttributeName.trim());
					}
					//all the domain values of a particular feature
					if(AttributeString.hasMoreTokens())
						AttributeDomain = AttributeString.nextToken();
					
					//stripping the period at the end of the line
					if (AttributeDomain.endsWith(".")) {
						AttributeDomain = AttributeDomain.substring(0, AttributeDomain.length() - 1);
					}
					// keeping track of continuous attributes
					if(AttributeDomain.trim().equals("continuous")) {
						continuousFeatureCount = continuousFeatureCount + 1;
						continuousFeatureIndex.add(FeaturesCount);
					}
					FeaturesCount = FeaturesCount + 1;
					
					StringTokenizer AttributeValToken = new StringTokenizer(AttributeDomain,",");
					ArrayList<String> DomainValues= new ArrayList<String>();
				
					AttributeValToken.countTokens();
					while (AttributeValToken.hasMoreTokens()) {
						String val =AttributeValToken.nextToken();					
						DomainValues.add(val.trim());
					}
					featureDomain.add(DomainValues);				
				
				}
			}	
			
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}			
	}
	public static  void loadData(String filename, ArrayList<ArrayList<String>> listname,int dataType) {
		/* Reads the training/testing data from the file into an list 
		 * of type ArrayList<ArrayList<String>>
		 * dataType -> 0 for training data
		 * 			-> 1 for testing data
		 * */
	
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line;
			
			while ((line = br.readLine()) != null) {
				if(line.length() > 0) { // to skip the blank lines at the end
					ArrayList<String> Values= new ArrayList<String>();
					StringTokenizer tokens = new StringTokenizer(line,",");
					int fIndex = 0;
					while (tokens.hasMoreTokens()) {
						String tokenVal = tokens.nextToken().trim();
						if( fIndex < FeaturesCount && dataType == 0 && !featureDomain.get(fIndex).contains("continuous") ) {
							if(!tokenVal.contains("?")) {
								int countIndex = featureDomain.get(fIndex).indexOf(tokenVal); // index of the attribute value in the domain values list
								ArrayList<Integer> tempList = valueCount.get(fIndex);
								int count = tempList.get(countIndex)+1;
								tempList.set(countIndex, count);
							}
						}
						Values.add(tokenVal);
						fIndex++;
					}									
					listname.add(Values);
				}
			}
			br.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}	

	private static HashMap<Integer,Integer> discretizeData(ArrayList<ArrayList<String>> trainingData) {
		/*calculates the best split for each continuous feature
		 * defines a Map datastructure for each feature
		 *  Map<Key,Value> : Key -> each value instance in the data
		 * 		 			 Value -> has three values at index 0 , 1 and 2
		 * 					index 0 - number of instances with target function value 1 
		 * 								for this particular Key
		 *   				index 1- number of instances with target function value 2
		 * 								for this particular Key
		 *  	 			index 2- number of instances for this particular Key
		
		 * */
		HashMap<Integer,Integer> continuousValSplits = new HashMap<Integer,Integer>();

		maxDomainvalue = new ArrayList<String>(FeaturesCount); // holds max value among the training data set for each feature
		for(int i=0;i<FeaturesCount;i++) {
			maxDomainvalue.add("");
			
		}
		for(int i=0;i<continuousFeatureCount;i++) {
			double maxInfoGain= 0;
			int chosenSplit = 0;
			
			//variables for entropy calculation(global)
			double totalvalues=0;
			double target1=0; 
			double target2=0;	
			double maxCount = 0;
			int maxValue = 0;
			
			int index = continuousFeatureIndex.get(i); // gets index for 
			Map<Integer,ArrayList<Double>> Map = new HashMap<Integer, ArrayList<Double>>(); 
			for(int dataIndex=0; dataIndex<trainingData.size();dataIndex++) {
				totalvalues = totalvalues+1;
				int key = Integer.parseInt(trainingData.get(dataIndex).get(index));
				String value =trainingData.get(dataIndex).get(FeaturesCount);
				
				ArrayList<Double> list;
				if(Map.containsKey(key)){
				    // if the key has already been used,
				    // we'll just grab the list and add the value to it
				    list = Map.get(key);
				    if(value.equals(targetFuntionValues.get(0))) {
						target1 = target1+1;
						double targetcount = list.get(0) + 1; 
						double localtotal = list.get(2) + 1;
						list.set(0,targetcount);
						list.set(2,localtotal);
					}
					else {
						target2 = target2+1;
						double targetcount = list.get(1) + 1; 
						double localtotal = list.get(2) + 1;
						list.set(1,targetcount);
						list.set(2,localtotal);
					}
				} 
				else {
				    // if the key hasn't been used yet,
				    // we'll create a new List<String> object, add the value
				    // and put it in the array list with the new key
				    list = new ArrayList<Double>();
				    if(value.equals(targetFuntionValues.get(0))) {
						target1 = target1+1;
						double targetcount = 1; 
						list.add(targetcount);
						list.add(0.0);
						list.add(1.0);

					}
					else {
						target2 = target2+1;
						double targetcount = 1; 
						list.add(0.0);
						list.add(targetcount);
						list.add(1.0);
					}
				    Map.put(key, list);
				}				
			}
			globalEntropy = calculateEntropy(target1,target2,totalvalues);
			Map<Integer,ArrayList<Double>> sortedMap = new TreeMap<Integer, ArrayList<Double>>(Map);  //sorting the Map 
			
			List<Integer> keyList = new ArrayList<Integer>(sortedMap.keySet());    // gives us all keys in the Map                                                       
			int splitcount = keyList.size();                 // sShould this be size()-1?
			for(int splitIndex =1;splitIndex < splitcount;splitIndex++) {     // splitting in order 
				//variables for entropy calculation for less than split value
				double lt_total=0;
				double lt_target1=0; 
				double lt_target2=0;
				double lt_entropy=0;
				//variables for entropy calculation for less than split value
				double gt_total=0;
				double gt_target1=0; 
				double gt_target2=0;
				double gt_entropy=0;
				double weightedEntropy=0;
				double infoGain = 0;
				int splitval=0;
				
				splitval =(keyList.get(splitIndex-1) + keyList.get(splitIndex))/2;				
		
				for (Integer key : sortedMap.keySet()) {
					ArrayList<Double> list = sortedMap.get(key);
					if(key <= splitval) {
						lt_total = list.get(2) +1;
						lt_target1= list.get(0)+ 1 ; 
						lt_target2= list.get(1) + 1;
					}
					else {
						gt_total = list.get(2) +1;
						gt_target1= list.get(0)+ 1 ; 
						gt_target2= list.get(1) + 1;
					}																									
	        }
		   // Calculating entropy and information gain for each split
			lt_entropy = calculateEntropy(lt_target1,lt_target2,lt_total);
			gt_entropy = calculateEntropy(gt_target1,gt_target2,gt_total);
			if((lt_total + gt_total) > 0) {
				weightedEntropy = ((lt_total * lt_entropy) +( gt_total * gt_entropy)) / (lt_total + gt_total);
			}
			infoGain = globalEntropy - weightedEntropy;
			 //Saving the split information with max info gain
			if(maxInfoGain < infoGain) {
				maxInfoGain = infoGain;
				chosenSplit = splitval;
			}
			}		
			//this is tracked to find the maximum domain value(for missing attribute values) 
			// in this case continuous attributes
			for (Integer key : sortedMap.keySet()) {
				ArrayList<Double> list = sortedMap.get(key);						
				Double count = list.get(2);
				if(count>maxCount) {
					maxValue = key;
					maxCount = count;
				}				
            }
			maxDomainvalue.set(index, String.valueOf(maxValue));			
			continuousValSplits.put(index, chosenSplit); // storing the split values in a map with the feature index as the key
			}		
		return continuousValSplits;
	}
	
	private static void findMaxDomainValue() {	
		//find the maximun value based on the count of the domain values stored
		for(int i=0;i<FeaturesCount;i++) {
			
			ArrayList<Integer> domainList = valueCount.get(i);
			int maxValue = Collections.max(domainList);
			int maxIndex = domainList.indexOf(maxValue);	
			ArrayList<String> domain = featureDomain.get(i);
			if(!domain.contains("continuous")){
					String maxDomainVal = domain.get(maxIndex);
					maxDomainvalue.set(i,maxDomainVal);
			}
		}	
	}
	
	
	public static void traverse(treeNode node)	{
		if(node.nodeType==1) {
			if(node.nodeLabel.equals(targetFuntionValues.get(0))) {
				target1++;
			}
			else if(node.nodeLabel.equals(targetFuntionValues.get(1)))	{
				target2++;
			}
		}
		else	{
			for(treeNode t:node.children) {
				traverse(t);
			}
		}
		
	}
	public static void pruning(treeNode node,treeNode parent,treeNode root1)	{		
		
		if(node.nodeType == 1) { // if its a leaf node , do nothing
			
		}
		else	{
			for(treeNode t:node.children)	{
				pruning(t,node,root1);    		//calling pruning for each child
			}
			Accuracy ob1=test(root1,validatingData); //finding accuracy before pruning of current node
			target1=0;target2=0;
			traverse(node);	//traverse method finds which target value is occurring most
			treeNode leaf=new treeNode();
			leaf.nodeType=1; leaf.nodeLabel=target1>=target2?(targetFuntionValues.get(0)):(targetFuntionValues.get(1));
			leaf.domainValue=node.domainValue;
			if(parent==null) {
				
				root1=leaf;
			}
			else
			parent.children.set(parent.children.indexOf(node), leaf);
			Accuracy ob2=test(root1,validatingData);	//finding accuracy after pruning
			if(ob1.accuracy>ob2.accuracy)	{
				if(parent==null)	root1=node;
				else
				parent.children.set(parent.children.indexOf(leaf), node);
			}
			
		}
		
	}
	private static ArrayList<treeNode> RandomForest(ArrayList<ArrayList<String>> Data, ArrayList<Integer> FeatureSet, int treeCount) {
		// Generates a treeCount number of trees using the random forest algorithm
		ArrayList<treeNode> forest = new ArrayList<treeNode>() ;
		int runCount  =treeCount;
		HashMap<Integer,Integer> continuousValSplits;
		continuousValSplits = discretizeData(Data); // calculating split values for continuous attributes
		for(int count = 0 ; count < runCount;count++) {
			//treeNode tree;
			ArrayList<Integer> featureSubset= getFeatureSubset(FeatureSet);
			//HashMap<Integer,Integer> continuousValSplits;
			//continuousValSplits = discretizeData(Data); // calculating split values for continuous attributes
			ArrayList<ArrayList<String>> randomDataSample = getRandomDatasetSample(trainingData);
			treeNode tree = train(randomDataSample,targetFuntionValues,featureSubset,1,FeatureSet,continuousValSplits);			
			forest.add(tree);
		}
		
		return forest;
	}	
	
	private static ArrayList<Integer> getFeatureSubset(ArrayList<Integer> featureSet) {
		/* This method randomly chooses a subset of features of size square root N
		 *  where N is the size of the remaining set of features
		 */
		int setSize = featureSet.size();
		int subsetSize = (int) Math.sqrt(setSize);
		ArrayList<Integer> subset = new ArrayList<Integer>();
		for(int i =0;i<subsetSize;i++) {
			Random random = new Random();			 
			int selectedVal = featureSet.get(random.nextInt(setSize)) ;			
			while(subset.contains(selectedVal)) {
				selectedVal = featureSet.get(random.nextInt(setSize)) ;
				}
			subset.add(selectedVal);
		}

		return subset;
	}
	private static ArrayList<ArrayList<String>> getRandomDatasetSample(ArrayList<ArrayList<String>> Data) {
		// This method randomly chooses N datapoints with replacement from Data of size N.
		int dataSize = Data.size();
		ArrayList<ArrayList<String>> sampledData = new ArrayList<ArrayList<String>>();
		for(int i = 0;i<dataSize;i++) {
			ArrayList<String> selectedDatapoint = new ArrayList<String>();
			Random random = new Random();
			selectedDatapoint = Data.get(random.nextInt(dataSize));
			sampledData.add(selectedDatapoint);			
		}		
		return sampledData;
	}
	
	private static Accuracy test(ArrayList<treeNode> randomForest, ArrayList<ArrayList<String>> data) {
		// This method tests the accuracy of random forest,using the given testing data
		Accuracy ob=new Accuracy();
		String target1Label = targetFuntionValues.get(0);
		String target2Label = targetFuntionValues.get(1);
		for(int dIndex=0;dIndex < data.size();dIndex++ ) {
			ArrayList<String> datapt = new ArrayList<String>();
			datapt = data.get(dIndex);
			String result;
			int tar1 = 0,tar2 = 0;
			for(int treeIndex=0;treeIndex <randomForest.size();treeIndex++) {
				treeNode root = randomForest.get(treeIndex);
				result = getResult(root,datapt);
				if(result != null) { // can be removed once missing attributes are handled

					if(result.equals(target1Label)) {
						tar1++;
					}
					else {
						tar2++;
					}
				}
			}
			if(tar1 > tar2) {
				result = target1Label;
			}
			else {
				result = target2Label;
			}
			
		if(result != null) {
				String actualResult = datapt.get(FeaturesCount);
				if (actualResult.endsWith(".")) {
					actualResult = actualResult.substring(0, actualResult.length() - 1);
				}
				if(actualResult.equals(target1Label)) {
					ob.actualTargetVal1 = ob.actualTargetVal1 +1;
					if(result.equals(actualResult)) {
						ob.trueTarget1 = ob.trueTarget1 +1;
					}else {
						ob.falseTarget2 = ob.falseTarget2 + 1;
					}				
				}else if(actualResult.equals(target2Label)) {
					ob.actualTargetVal2 = ob.actualTargetVal2 + 1;
					if(result.equals(actualResult)) {
						ob.trueTarget2 = ob.trueTarget2 +1;
					}else {
						ob.falseTarget1 = ob.falseTarget1 + 1;
					}				
				}
			
		  }
		}
		int temp=(ob.trueTarget1+ob.trueTarget2+ob.falseTarget1+ob.falseTarget2);
		if(temp==0) ob.accuracy=0;
		else
		ob.accuracy=(float)(ob.trueTarget1+ob.trueTarget2)/temp;
		return ob;
	}
	

	public static void main(String[] args) {
		/* intial file processing */
		
		System.out.println("*****Loading Arrributes*****\n");
		loadFeatures(featureFile);
		// doing this to keep track of max occuring val of a feature to handle missing atrributes
		for(int i=0;i<FeaturesCount;i++) {
			int count = featureDomain.get(i).size();
			ArrayList<Integer> countList = new ArrayList<Integer>();
			for(int j=0;j<count;j++) {
				countList.add(0);
			}
			valueCount.add(countList);
		}
		
		System.out.println("*****Loading Traing Data in File*****\n");
		loadData(trainingDataFilename,trainingData,0);			//reading the training data
		HashMap<Integer,Integer> continuousValSplits = null;
		if(continuousFeatureCount!=0) {
			System.out.println("*****Discretizing Continous Attributes*****\n");
			continuousValSplits = discretizeData(trainingData); // calculating split values for continuous attributes
		}
		
		findMaxDomainValue();
		//holds indices of features whose entropy needs to be calculated
		ArrayList<Integer> AttributeSet = new ArrayList<Integer>();
		for(int i=0;i<FeaturesCount;i++) {
			AttributeSet.add(i);
		}
		
		//Task1 - building decision tree using ID3 algorithm
		//learn the decision tree
		System.out.println("\n*****Building Decision Tree through ID3*****\n");
		long startTime = System.currentTimeMillis();
		treeNode root = train(trainingData,targetFuntionValues,AttributeSet,0,AttributeSet,continuousValSplits);
		long duration = (System.currentTimeMillis()-startTime);	
		double decisionTreeTrainingTime = (double)duration / 1000; 
		//test the decision tree
		System.out.println("*****Testing ID3*****\n");
		loadData(testingDataFilename,testingData,1); // getting test data
		Accuracy ID3=test(root,testingData);		
		
		//Task 2 - Reduced Error Pruning
		System.out.println("\n*****Reduced Error Pruning*****\n\n");
		trainData2=new ArrayList<ArrayList<String>> (trainingData.subList(0,(int) (trainingData.size()*0.7)));
		validatingData=new ArrayList<ArrayList<String>> (trainingData.subList((int) (trainingData.size()*0.7),(trainingData.size())));
		//System.out.println("Size of Training data: "+trainingData.size());
		//System.out.println("Size of Test Data : "+testingData.size());
		//System.out.println("size of training data for pruning:"+trainData2.size());
		//System.out.println("Size of validting data is:"+validatingData.size());
		startTime = System.currentTimeMillis();
		treeNode root1=train(trainData2,targetFuntionValues,AttributeSet,0,AttributeSet,continuousValSplits);
		pruning(root1,null,root1);
		duration = (System.currentTimeMillis()-startTime);	
		double pruningTrainingTime= (double)duration/1000;		
		//Testing after pruning
		System.out.println("\n*****Testing after Pruning*****\n");
		Accuracy pruning=test(root1,testingData);		
		
		//Task3 - Random Forest Algorithm
		System.out.println("\n****Random Forest*****\n");
		startTime = System.currentTimeMillis();
		ArrayList<treeNode> randomForest = RandomForest(trainingData,AttributeSet,5);     //running random forest for specific number of trees
		duration = (System.currentTimeMillis()-startTime);	
		double randomForestTrainingTime= (double)duration/1000;
		//testing random forest
		System.out.println("\n****testing Random Forest*****\n");
		Accuracy randomForestPerformance = test(randomForest,testingData);	
		
		//Analyse and output the results
		System.out.println("*******************************************************RESULTS*******************************************************");
		System.out.println(" ");
		System.out.println("------------------------------------------------------------------------------------------------------------------- ");
		System.out.println("Performance Measure\t|\tID3 \t|Pruning\t| Random forest\t|");
		System.out.println("------------------------------------------------------------------------------------------------------------------- ");
		//System.out.println("Training time\t\t|"+decisionTreeTrainingTime+"Seconds\t|"+pruningTrainingTime+"Seconds\t|"+randomForestTrainingTime+"Seconds\t");
		System.out.print("Training time\t\t|");
		//System.out.print(decisionTreeTrainingTime);
		System.out.printf("%.2f",decisionTreeTrainingTime);
		System.out.print(" Seconds\t|");
		System.out.printf("%.2f",pruningTrainingTime);
		System.out.print(" Seconds\t|");
		System.out.printf("%.2f",randomForestTrainingTime);
		System.out.print(" Seconds\t|");
		System.out.println(" ");
		System.out.println("Accuracy\t\t|"+ID3.accuracy*100+"%\t|"+pruning.accuracy*100+"%\t|"+randomForestPerformance.accuracy*100+"%\t|");
		System.out.println(" ");
		System.out.println("For target function value ->  >50k: ");
		System.out.println(" ");
		System.out.println("Precision\t\t|"+ID3.calculatePrecision(0)+"\t|"+pruning.calculatePrecision(0)+"\t|"+randomForestPerformance.calculatePrecision(0)+"\t|");
		System.out.println(" ");
		System.out.println("Recall\t\t\t|"+ID3.calculateRecall(0)+"\t|"+pruning.calculateRecall(0)+"\t|"+randomForestPerformance.calculateRecall(0)+"\t|");
		System.out.println(" ");
		System.out.println("FMeasure\t\t|"+ID3.calculatefMeasure(0)+"\t|"+pruning.calculatefMeasure(0)+"\t|"+randomForestPerformance.calculatefMeasure(0)+"\t|");
		System.out.println(" ");
		System.out.println("For target funtion value ->  <=50k: ");
		System.out.println(" ");
		System.out.println("Precision\t\t|"+ID3.calculatePrecision(1)+"\t|"+pruning.calculatePrecision(1)+"\t|"+randomForestPerformance.calculatePrecision(1)+"\t|");
		System.out.println(" ");
		System.out.println("Recall\t\t\t|"+ID3.calculateRecall(1)+"\t|"+pruning.calculateRecall(1)+"\t|"+randomForestPerformance.calculateRecall(1)+"\t|");
		System.out.println(" ");
		System.out.println("FMeasure\t\t|"+ID3.calculatefMeasure(1)+"\t|"+pruning.calculatefMeasure(1)+"\t|"+randomForestPerformance.calculatefMeasure(1)+"\t|");

	}
}


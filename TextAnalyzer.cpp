// TextAnalyzer.cpp : Defines the entry point for the application.
//

#include "TextAnalyzer.h"

using namespace std;

/*
long hash(char* key) {
	long hashVal = 0;
	while (*key != '/0') {
		hashVal = (hashVal << 4) + *(key++);
		long g = hashVal & 0xF0000000L;
		if (g != 0) hashVal ^= g >> > 24;
		hashVal &= ~g;
	}
	return hashVal;
}
*/

bool pairCompare(pair<string,int>& a, pair<string,int>& b) {
	return a.second > b.second;
}

//input stemmed document and output top 50 stems: stem,freq,rank. Followed by unique,total word count
void freqSorter(string fileName) {
	ifstream fileIn(fileName);
	vector<int> uniqueWordCount;
	int totalWordCount = 0;
	unordered_map<string, int> stemHash;
	string stem;

	while (fileIn >> stem) {
		totalWordCount++;
		if (stemHash.find(stem) == stemHash.end()) uniqueWordCount.push_back(totalWordCount);
		stemHash[stem]++;
	}
	fileIn.close();

	vector<pair<string, int>> stemFreq(stemHash.begin(), stemHash.end());
	sort(stemFreq.begin(), stemFreq.end(), pairCompare);
	ofstream  fileOut;
	string fileOutput = fileName.substr(0,fileName.size()-4) + "_Output.csv";
	fileOut.open(fileOutput);
	fileOut << '"' << "Stem" << '"' << ", " << '"' << "Frequency" << '"' << ", " << '"' << "Rank" << '"' << ", " << '"' << "Probability" << '"' << '\n';
	float twc = totalWordCount;
	for (int i = 0; i < stemFreq.size() && i < 50; i++) fileOut << '"' << stemFreq[i].first << '"' << ", " << stemFreq[i].second << ", " << i + 1 << ", " << stemFreq[i].second/twc << '\n';
	fileOut << '\n' << '"' << "Total World Count" << '"' << ", " << '"' << "Unique World Count" << '"' << '\n';
	for (int i = 0; i < uniqueWordCount.size(); i++) fileOut << uniqueWordCount[i] << ", " << i+1 << '\n';
	fileOut.close();
}

int main(int argc, char* argv[])
{
	for (int i = 1; i < argc; i++) freqSorter(argv[i]);
	return 0;
}

#include "str_ptr_match.h"

using namespace std;

int main()
{
	ifstream infile1;
	infile1.open("train_srh_all.txt");
	string line,word,line_t;
	
	//vector<string> train_vec = {"K85","K326","K648","K776","K1096","K1172","T170","Z36","Z202"};
	vector<string> train_vec = { "K85", "K230", "K301", "K302", "K325", "K326", "K511", "K642", "K648", "K776", "K795", "K827", "K837", "K841", "K1017", "K1096", "K1172", "K1234", "K9023", "K9046", "K9050", "K9067", "K9602", \
		"K9454", "T170", "T8350", "T8351", "T8352", "Z36", "Z112", "Z201", "Z202", "C7670", "C7683", "D2819", "D3012", "G96", "G1014"};
	unordered_map <string, vector<string> >train_map;
	while (getline(infile1, line))
	{
		vector<string> line_s;
		istringstream record(line);
		record >> line_t;
		//cout << "each line is " << line << endl;
		while (record >> word)
			line_s.push_back(word);
		train_map[line_t] = line_s;
		//line_s.clear();
	}
	//const int SIZE = 10;
	//string str_src[10] = {"guangzhou", "guangzhoudong", "dongguandong", "huizhou", "heyuan", "longchuang", "ganzhou", "jian", "nanchangxi", "jiujiang" };
	string train_input;
	string str_match;	
	cout << "please input train string: ";
	cin >> train_input;
	int str_train_idx = str_ptr_match(train_vec, train_input, train_vec.size());
	if (str_train_idx == -1)
		cout << "please input a right train string" << endl;
	else
	{
		train_input = train_vec[str_train_idx];
		cout << "train input after correction is " << train_input << endl;
	}
	cout << "please input station string: ";
	while (cin >> str_match)
	{
		int str_idx = str_ptr_match(train_map[train_input], str_match, train_map[train_input].size());			//index of the source string
		if (str_idx == -1)
			cout << "no matching" << endl;
		else
		{
			cout << "matching success, the result is: ";
			cout << train_map[train_input][str_idx] << endl;
		}
		cout << "please input station string: ";
	}
	cout << "over" << endl;
	infile1.close();
	return 0;
}
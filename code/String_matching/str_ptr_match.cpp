#include "str_ptr_match.h"
using namespace std;

int str_ptr_match(const vector<string>& str_src, const string str_match,int n)
{
	bool debug = true;
	//select the max value of k string
	double count_ratio_max = -1;
	int idx_dst = -1;

	for (int k = 0; k < n; k++)
	{
		//print matching result of each string 
		string str1 = str_src[k];
		string str_dst = "";
		int count_mtc = 0;
		int idx_mtc = -1;
		int idx_inc = 2;
		for (int i = 0; i < str_match.length(); i++)
		{
			int idx_end = idx_mtc + idx_inc;
			bool mtc_flag = false;
			for (int j = idx_mtc + 1; j <= idx_end && j < str1.length(); j++)
			{
				if (str_match[i] == str1[j])
				{
					idx_mtc = j;
					count_mtc++;
					str_dst += str1[j];
					mtc_flag = true;
					break;
				}
			}
			if (mtc_flag)
				idx_inc = 2;
			else
				idx_inc++;
		}
		if (debug)
			cout << "original string is : " << str1 << "  matching string is : " << str_dst << "  matching number is : " << count_mtc;

		//find the max string and max matching value
		double count_ratio_tmp1 = double(count_mtc) / str1.length();
		double count_ratio_tmp2 = double(abs(int(str_match.length() - count_mtc))) / str_match.length();
		double count_ratio_tmp = count_ratio_tmp1 - count_ratio_tmp2;
		if (debug)
			cout << "  matching ratio is ;" << count_ratio_tmp;
		//cout << "  matching ratio1 is ;" << count_ratio_tmp1;
		//cout << "  matching ratio2 is ;" << count_ratio_tmp2;
		if (debug)
			cout << endl;
		if (count_ratio_max < count_ratio_tmp)
		{
			idx_dst = k;
			count_ratio_max = count_ratio_tmp;
		}

		//when one station is a subset of another station
		if ((count_ratio_max == count_ratio_tmp) && (idx_dst>-1))
		{
			if (str_src[idx_dst].length() < str_src[k].length())
				idx_dst = k;	
		}
	}

	//when picture adjustment is wrong ,we need to take another picture
	if (count_ratio_max < -0.2)
	{
		cout << "recognition is wrong ,please take another picture!" << endl;
		return -1;
	}
		
	cout << "dst string is : " << str_src[idx_dst] << "  dst number is : " << idx_dst << "  ratio is : " << count_ratio_max << endl;

	return idx_dst;
}
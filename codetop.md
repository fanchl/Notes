## 322 零钱兑换
给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。  
计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。  
你可以认为每种硬币的数量是无限的。

**示例 1：**
```
输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```
**示例 2：**
```
输入：coins = [2], amount = 3
输出：-1
```
**示例 3：**
```
输入：coins = [1], amount = 0
输出：0
```

**提示：**
`1 <= coins.length <= 12`
`1 <= coins[i] <= 231 - 1`
`0 <= amount <= 104`

> 动态规划
维护一个 dp 数组，存储组成该金额需要的最少硬币数。  
这里设置最大值不能直接用 INT_MAX, 要不 +1 之后会溢出，成为负数，导致之后的最小值都出错了。
```C++
class Solution {
public:
	int coinChange(vector<int>& coins, int amount) {
		int MAX = amount + 1;
		vector<int> dp(amount + 1, MAX);
		dp[0] = 0;
		for (int i = 1; i <= amount; i++){
			for (int coin : coins){
				if (i - coin >= 0){
					dp[i] = min(dp[i], dp[i - coin] + 1);
				}
				
			}
		}
		return dp[amount] == MAX ? -1 : dp[amount];
	}
};
```

## 15 三数之和
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

**示例 1：**
```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
```
**示例 2：**
```
输入：nums = []
输出：[]
```
**示例 3：**
```
输入：nums = [0]
输出：[]
```


> 思路：排序+双指针 

固定左边界，移动剩下的两个指针。
需要排除相同的组合，遍历的时候如果遇到相同的值的时候，直接跳过。
```C++
class Solution {
public:
	vector<vector<int>> threeSum(vector<int>& nums) {
		sort(nums.begin(), nums.end());
		vector<vector<int>> res;
		int n = nums.size() - 2;
		for (int k = 0; k < n; k++) {
			if (nums[k] > 0) break;
			if (k > 0 && nums[k] == nums[k - 1]) continue;
			int i = k + 1, j = nums.size() - 1;
			while (i < j) {
				int sum = nums[k] + nums[i] + nums[j];
				if (sum < 0) {
					i++;
					while (i < j && nums[i] == nums[i - 1]) i++;
				} else if (sum > 0) {
					j--;
					while (i < j && nums[j] == nums[j + 1]) j--;
				} else {
					res.push_back(vector<int> {nums[k], nums[i], nums[j]});
					i++;
					j--;
					while (i < j && nums[i] == nums[i - 1]) i++;
					while (i < j && nums[j] == nums[j + 1]) j--;
				}
			}
		}
		return res;
	}
};
```


## 5 最长回文字符串
给你一个字符串 s，找到 s 中最长的回文子串。

**示例 1：**
```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```
***示例 2：**
```
输入：s = "cbbd"
输出："bb"
```

> 思路：中心扩散算法

+ C++ `pair<T, T>` 的用法。
+ 回文字符串 的长度有可能是奇数也有可能是偶数，两者需要分别考虑。

```C++
class Solution {
public:
    string longestPalindrome(string s) {
        int len = s.size();
        if (len == 0 || len == 1) return s;
        int start = 0;
        int end = 0;
        for (int i = 0; i < len; i++) {
            auto [l1, r1] = expendAroundCenter(s, i, i);
            auto [l2, r2] = expendAroundCenter(s, i, i + 1);
            if (r1 - l1 > end - start) {
                start = l1;
                end = r1;
            }
            if (r2 - l2 > end - start) {
                start = l2;
                end = r2;
            }
        }
        return s.substr(start, end - start + 1);
    }

    pair<int, int> expendAroundCenter(string s, int l, int r) {
        while (l >= 0 && r < s.size() && s[r] == s[l]) {
            l--;
            r++;
        }
        return {l + 1, r - 1};
    }
};
```
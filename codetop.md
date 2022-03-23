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

## 72 编辑距离
给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。

你可以对一个单词进行如下三种操作：

+ 插入一个字符  
+ 删除一个字符  
+ 替换一个字符  

**示例 1：**
```
输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```
**示例 2：**
```
输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

> 思路：动态规划  

注意初始化状态，以及状态如何转移。

```C++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size();
        int m = word2.size();
        if (n * m == 0) return n + m;
        vector<vector<int>> dp (n + 1, vector<int> (m + 1));

        for (int i = 0; i <= n; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= m; j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                int left = dp[i - 1][j];
                int down = dp[i][j - 1];
                int left_down = dp[i - 1][j - 1];
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = min(left_down, min(left + 1, down + 1));
                    // dp[i][j] = left_down;
                } else {
                    dp[i][j] = min(left_down + 1, min(left + 1, down + 1));
                }
            }
        }
        return dp[n][m];
    }
};
```

## 206 反转链表
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

**示例 1：**  
![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex1.jpg)
```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```
**示例 2：**  
![](https://assets.leetcode.com/uploads/2021/02/19/rev1ex2.jpg)
```
输入：head = [1,2]
输出：[2,1]
```
**示例 3：**  
```
输入：head = []
输出：[]
```

> 思路：迭代

每次对某变量赋值之前先将其赋值给另一变量。

```C++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while (cur) {
            ListNode* next = cur -> next;
            cur -> next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
};
```

## 200 岛屿数量
给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

**示例 1：**  
```
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

**示例 2：**  
```
输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3
```

> 思路：dfs

思路和剑指offer第12题“矩阵中的路径”解法类似。

```C++
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int count = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i ,j);
                    count++;
                }
            }
        }
        return count;
    }

    void dfs(vector<vector<char>>& grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.size() || j >= grid[0].size() || grid[i][j] == '0') return;
        grid[i][j] = '0';
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }
};
```


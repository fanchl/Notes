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
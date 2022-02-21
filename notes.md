# 剑指offer

## C++ 基础

### 常用函数

max min pow

### 链表

```C++ 
ListNode* res = new ListNode(0), *cur = res;
```

### 类

**new创建类对象与不用new区别****

下面是总结的一些关于new创建类对象特点：

- new创建类对象需要指针接收，一处初始化，多处使用
- new创建类对象使用完需delete销毁
- new创建对象直接使用堆空间，而局部不用new定义类对象则使用栈空间
- new对象指针用途广泛，比如作为函数返回值、函数参数等
- 频繁调用场合并不适合new，就像new申请和释放内存一样

### map

**初始化**

新建 map 的时候可以不初始化。比如 `map<int, bool> exist;`

**判断是否存在某个 key**

直接可以使用 `if (exs[num])`

**遍历 map**

```C++
map<int, int>::iterator iter = m.begin();
while (iter != m.end()){
    cout << iter -> first << ":" << iter -> second << endl;
    iter++;
}
```



### vector

**初始化一维 vector**

```C++
vector<string> words1 {"the", "frogurt", "is", "also", "cursed"};
vector<int> a(10, 0))
```

**初始化二维 vector**

```C++
vector<vector<int>> v(20, vector<int>(10, 0));
```

#### push_back

`v.push_back(10)` 在 vector `v` 后面添加元素 10

### string

#### erase

`s.erase(i)` 删除 `i` 位置之后的所有字符

`s.erase(i, n)` 从第 `i` 个位置删除多少字符

```c++
std::string s = "This Is An Example";
std::cout << "1) " << s << '\n';
 
s.erase(7, 3); // erases " An" using overload (1)
std::cout << "2) " << s << '\n';

1) This Is An Example
2) This Is Example
```



#### insert

`s.insert(i, "hello")` 在第 `i` 个位置之后添加字符串 `"hello"`



#### append

`s.append(3, 'a') ` 在字符串 `s` 之后添加 3 个字符 `'a'`

`s.append("hello")` 在字符串 `s` 后添加字符串 `"hello"`



### stack

#### Element access

+ `s.top`

#### Capacity

+ `s.empty()`
+ `s.size()`

#### Modifiers

+ `s.push()`
+ `s.pop()` 返回值为空，需要用 `s.top()` 来获取栈顶元素。



## 03 数组中重复的数字

在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

**示例 1：**

```
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 
```

> 思路1：如果所有数字都只出现了一次，那么每个值都能找到一一对应的索引。每个值既要判断是否等于自己的索引，本身也可以作为索引去调换。
>
> 遍历数组：
>
> + 如果该值等于索引值，那么 continue
> + 如果该值不等于索引值，那么准备将该值调换到等于索引值的位置（将该值作为索引值），先判断是否与要调换位置的值相等？
>   + 如果相等，那么确实存在重复值，返回该重复值即可
>   + 如果不相等，则可以进行调换，即 swap(nums[i], nums[nums[i]])。
>
> ```C++
> class Solution {
> public:
>     int findRepeatNumber(vector<int>& nums) {
>         int res = -1;
>         if (nums.size() < 2) return -1;
>         int i = 0;
>         while (i < nums.size()){
>             if (nums[i] == i){
>                 i++;
>                 continue;
>             } else if (nums[i] == nums[nums[i]]){
>                 return nums[i];
>             } else{
>                 swap(nums[i], nums[nums[i]]);
>             }
>         }
>         return -1;
>     }
> };
> ```
>
> 
>
> 思路2: 创建 hashmap，遍历数组，存入map，如果存在该数字的记录，则证明该数字重复。
>
> ```C++
> class Solution {
> public:
>     int findRepeatNumber(vector<int>& nums) {
>         int res = -1;
>         if (nums.size() < 2) return -1;
>         map<int, bool> exs;
>         for (int num: nums){
>             if (exs[num]) return num;
>             else exs[num] = true;
>         }
>         return -1;
>     }
> };
> ```
>
> 





## 04 二维数组中的查找

在一个 `n * m` 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

**示例:**

```
现有矩阵 matrix 如下：
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]

```

给定 `target = 5`，返回 `true`。

给定 `target = 20`，返回 `false`。



> 思路：暴力法未利用矩阵 **“从上到下递增、从左到右递增”** 的特点，显然不是最优解法。
>
> 注意到两个位置：**左下角**和**右上角**
>
> 对于左下角来说，往上走都是小于它的，往右走都是大于它的。类似于二叉搜索树。同理可以推断右上角的响应规则。
>
> 将左下角的值作为 flag，
>
> + 如果 `target > flag`，那么 target 一定不在最左边这一列，则可以消除最左边这一列 (j++)
> + 如果  `target < flag`，那么 target 一定不在最下面这一行，则可以消除最下面这一行 (i--)
>
> ```C++
> class Solution {
> public:
>     bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
>         if (matrix.size() == 0 || matrix[0].size() == 0) return false;
>         int i = matrix.size() - 1;
>         int j = 0;
>         while (i >= 0 && j < matrix[0].size()){
>             int flag = matrix[i][j];
>             if (target > flag){
>                 j++;
>             } else if (target < flag){
>                 i--;
>             } else{
>                 return true;
>             }
>         }
>         return false;
>     }
> };
> ```



## 05 替换空格

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

**示例 1：**

```
输入：s = "We are happy."
输出："We%20are%20happy."
```

> 思路：从头开始替换会出现索引变化，从结尾开始替换。
>
> 遍历数组，遇到空格，删除空格并添加添加 "%20".
>
> ```C++
> class Solution {
> public:
>     string replaceSpace(string s) {
>         for (int i = s.size() - 1; i >= 0; i--){
>             if (s[i] == ' '){
>                 s.erase(i, 1);
>                 s.insert(i, "%20");
>             }
>         }
>         return s;
>     }
> };
> ```



## 06 从尾到头打印链表

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

**示例 1：**

```
输入：head = [1,3,2]
输出：[2,3,1]
```

> 思路：遍历链表的时候，使用数组保存，最后使用 reverse 函数对数组实现反转，实现从尾到头打印链表。
>
> ```c++
> class Solution {
> public:
>     vector<int> reversePrint(ListNode* head) {
>         vector<int> res;
>         while (head != NULL){
>             res.push_back(head -> val);
>             head = head -> next;
>         }
>         reverse(res.begin(), res.end());
>         return res;
>     }
> };
> ```



## 07 重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。

假设输入的前序遍历和中序遍历的结果中**都不含重复的数字**。

 <img src="https://assets.leetcode.com/uploads/2021/02/19/tree.jpg" style="zoom:50%;" />

**示例 1:**

```
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
```

**示例 2:**

```
Input: preorder = [-1], inorder = [-1]
Output: [-1]
```

> 思路：
>
> 对每一棵（子）树，都能通过在中序遍历中，基于根节点的位置，将其划分成左右子树。然后接着对每一棵左右子树进行相同的划分操作。
>
> 通过中序遍历数组来确定树的根节点位置，左右边界位置。
> 设根节点在中序遍历数组中索引为 `i`，在前序遍历数组中的索引为 `root`，在划分左右子树之后，
>
> + 那么左子树的左右边界分别为大树的左边界到 `i-1`，左子树的根节点为 `root + 1`，；
>
> + 右子树的左右边界分别为 `i+1` 和大树的右边界，右子树的根节点在前序遍历数组中索引为 `根节点索引+左子树长度+1`，可表示为 `root + (i - left) + 1`。
>
> ```c++
> /**
>  * Definition for a binary tree node.
>  * struct TreeNode {
>  *     int val;
>  *     TreeNode *left;
>  *     TreeNode *right;
>  *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
>  * };
>  */
> class Solution {
> public:
>     TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
>         this -> preorder =  preorder;
>         for (int i = 0; i < inorder.size(); i++){
>             in_idx[inorder[i]] = i;
>         }
>         return constructCore(0, 0, inorder.size() - 1);
>     }
> 
> private:
>     vector<int> preorder;
>     unordered_map<int, int> in_idx;
>     TreeNode* constructCore(int root, int left, int right){
>         if (left > right) return NULL;
>         TreeNode* node = new TreeNode(preorder[root]);
>         int i = in_idx[preorder[root]];  // 通过根节点在前序遍历数组中的索引找到在中序遍历数组中的索引
>         node -> left = constructCore(root + 1, left, i - 1);  // 构建左子树
>         node -> right = constructCore(root + i - left + 1, i + 1, right);  // 构建右子树
>         return node;
>     }
> };
> ```



## 09 用两个栈实现一个队列

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

**示例 1：**

```
输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]
```

**示例 2：**

```
输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]
```

> 思路：使用两个栈 A、B 来实现一个队列的功能。可以分别用 A 来实现 `appendTail` 和用 B 来实现 `deleteHead`。
>
> + 添加到尾部操作比较简单，直接添加到 A 的尾部即可；
> + 进行删除头部操作的时候，需要注意到 B 可能是空的，A 也可能是空的，需要分情况讨论。
>
> ```C++
> class CQueue {
> public:
>     stack<int> A;
>     stack<int> B;
>     CQueue() {}
>     
>     void appendTail(int value) {
>         A.push(value);
>     }
>     
>     int deleteHead() {
>         if (!B.empty()){
>             int head = B.top();
>             B.pop();
>             return head;
>         } else if (!A.empty()){
>             while (!A.empty()){
>                 B.push(A.top());
>                 A.pop();
>             }
>             int head = B.top();
>             B.pop();
>             return head;
>         } else{
>             return -1;
>         }
>     }
> };
> ```



## 10-1 斐波那契数列

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：

```
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```

斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**示例 1：**

```
输入：n = 2
输出：1
```

**示例 2：**

```
输入：n = 5
输出：5
```

> 思路：直接递推公式求出每一个 F(N)
>
> ```c++
> class Solution {
> public:
>     int fib(int n) {
>         if (n < 0) return -1;
>         if (n == 0) return 0;
>         if (n == 1) return 1;
>         int p = 0, q = 0, r = 1;
>         for (int i = 2; i <= n; i++){
>             p = q;
>             q = r;
>             r = (p + q) % 1000000007;
>         }
>         return r;
>     }
> };
> ```



## 10-2 青蛙跳台阶问题

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**示例 1：**

```
输入：n = 2
输出：2
```

**示例 2：**

```
输入：n = 7
输出：21
```

**示例 3：**

```
输入：n = 0
输出：1
```

> 思路：
>
> 这是一个斐波那契数列问题。
>
> ```C++
> class Solution {
> public:
>     int numWays(int n) {
>         if (n == 0) return 1;
>         if (n == 1) return 1;
>         if (n == 2) return 2;
>         int p = 1, q = 1, r = 2; 
>         for (int i = 3; i <= n; i++){
>             p = q;
>             q = r;
>             r = (p + q) % 1000000007;
>         }
>         return r;
>     }
> };
> ```



## 11 旋转数组的最小数字

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。

给你一个**可能存在重复元素值**的数组 numbers ，它原来是一个升序排列的数组，并按上述情形进行了一次旋转。请返回旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一次旋转，该数组的最小值为1。  

**示例 1：**

```
输入：[3,4,5,1,2]
输出：1
```

**示例 2：**

```
输入：[2,2,2,0,1]
输出：0
输入：[1,1,1,0,1]
输出：0
输入：[1,0,1,1,1]
输出：0
```

> 思路：
>
> 旋转数组，分析知道这其实是两个有序数组，而且这两个有序数组还存在一个关系，即右边的有序数组的值小于等于左边的有序数组的值。
>
> 对索引使用二分法，初始 left 和 right 分别为数组的边界。
>
> 判断 a = nums[mid] 和 b = nums[right] 之间的大小关系。
>
> + a > b：a 位于左侧有序数组，旋转点在 a 右边，那么可以确定新的左边界 left = mid + 1 （需+1，否则可能陷入无限循环）；
> + a < b：a 位于右侧有序数组，旋转点在 a 左边（也有可能是 a），那么可以确定新的右边界 right = mid （不能+1，因为有可能 a 就是旋转点）；
> + a = b：这种情况，无法直接判断 a 具体在哪边的有序数组，因为该数组**可能存在重复元素**。因为是为了找到最小值（旋转点的值，目前），因此可以缩小一个数的范围，确定新的右边界：令 right--。
>
> ```C++
> class Solution {
> public:
>     int minArray(vector<int>& numbers) {
>         // 首先考虑特殊情况，加入没有旋转，即旋转点为第一个元素
>         if (numbers.size() <= 0 ) return -1;
>         if (numbers[numbers.size() - 1] > numbers[0]) return numbers[0];
>         int left = 0;
>         int right = numbers.size() -1;
>         while (left < right){
>             int mid = (left + right) / 2;
>             if (numbers[mid] < numbers[right]){
>                 right = mid;
>             } else if(numbers[mid] > numbers[right]){
>                 left = mid + 1;
>             } else{
>                 right--;
>             }
>         }
>         return numbers[left];
>     }
> };
> ```



## 12 矩阵中的路径

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

例如，在下面的 3×4 的矩阵中包含单词 "ABCCED"（单词中的字母已标出）。

![](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)

**示例 1：**

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

**示例 2：**

```
输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
```

> 思路：
>
> 使用回溯法，深度优先搜索。
>
> **DFS 解析**
>
> + 递归参数：当前元素在 borad 中的行列索引 i 和 j，当前目标字符在 word 中的索引 k。
> + 终止条件：
>   + 返回 false：
>     + 行列索引越界
>     + 当前矩阵元素和目标字符不同
>     + 当前矩阵元素已经访问过 （可合并到上一条）
>   + 返回 true：
>     + k = word.size() -1，即字符串已经全部匹配成功。
>
> ```C++
> class Solution {
> public:
>     bool exist(vector<vector<char>>& board, string word) {
>         rows = board.size();
>         cols = board[0].size();
>         for (int i = 0; i < rows; i++){
>             for (int j = 0; j < cols; j++){
>                 if (dfs(board, word, i, j, 0)){
>                     return true;
>                 }
>             }
>         }
>         return false;
>     }
> 
> private:
>     int rows, cols;
>     bool dfs(vector<vector<char>>& board, string word, int i, int j, int k){
>         if (i >= rows || i < 0 || j >= cols || j < 0 || board[i][j] != word[k]){
>             return false;
>         }
>         if (k == word.size() - 1){
>             return true;
>         } 
>         board[i][j] = '\0';
>         bool res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k + 1) || dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i, j - 1, k + 1);
>         board[i][j] = word[k];
>         return res;
>     }
> };
> ```



## 13 机器人的运动范围

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

**示例 1：**

```
输入：m = 2, n = 3, k = 1
输出：3
```

**示例 2：**

```
输入：m = 3, n = 1, k = 0
输出：1
```

> 思路：
>
> 使用 DFS，12 题中的出发点是任意的，移动方向包括上下左右四种，且是需要找到是否存在一条路径。
>
> 该题出发点固定，且在(0,0)，可以固定成两种方向移动，计算能够访问区域的面积。需要设置 visited。
>
> <img src="https://pic.leetcode-cn.com/1603024999-iYtADx-Picture21.png" style="zoom: 33%;" />
>
> ```C++
> class Solution {
> public:
>     int movingCount(int m, int n, int k) {
>         vector<vector<bool>> visited (m, vector<bool>(n, false));
>         int res = dfs(visited, 0, 0, m, n, k);
>         return res;
>     }
> 
> private:
>     int dfs(vector<vector<bool>>& visited, int i, int j, int m, int n, int k){
>         if (i >= m || j >= n || visited[i][j] || bitsum(i) + bitsum(j) > k){
>             return 0;
>         }
>         visited[i][j] = true;
>         return 1 + dfs(visited, i + 1, j, m, n, k) + dfs(visited, i, j + 1, m, n, k);
>     }
> 
>     int bitsum(int n){
>         int sum = 0;
>         while(n > 0){
>             sum += n % 10;
>             n /= 10;
>         }
>         return sum;
>     }
> 
> };
> ```



## 14-1 剪绳子

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]\*k[1]\*...\*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

**示例 1：**

```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
```

**示例 2:**

```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```

> 思路1：
>
> **动态规划：**
>
> 1. 想要长度为 n 的绳子剪掉后的最大乘积，可以从前面比 n 更小的绳子转移而来。
> 2. 用一个 dp 数组记录从 0 到 n 长度的绳子剪掉之后的最大乘积，也就是 dp[i] 表示长度为 i 的绳子剪成 m 段后的最大乘积，初始化 dp[0]=0, dp[1]=0, dp[2]=1
> 3. 剪了一段之后，剩下 (i-j) 长度可以剪（`dp[i-j] * j`）也可以不剪（`(i-j)*j`）。
> 4. 第一段的长度可以是 [1,i)，对所有 `j` 不同的情况取最大值，因此最终的 dp[i] 转移方程为 `dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))`
>
> ```C++
> class Solution {
> public:
>     int cuttingRope(int n) {
>         vector<int> dp (n + 1, 0);
>         dp[2] = 1;
>         for (int i = 3; i <= n; i++){
>             for (int j = 1; j < i; j++){
>                 dp[i] = max(dp[i], max((i - j) * j, (dp[i - j] * j)));
>             }
>         }
>         return dp[n];
>     }
> };
> ```
>
> **贪心算法**：
>
> 尽可能把绳子分成长度为 3 的小段，这样乘积最大
>
> *切分规则*
>
> + 最优：3。将绳子尽可能切分成多个长度为 3 的片段，留下的最后一段绳子的长度为 0,1,2 三种情况。
> + 次优：2。若最后一段绳子长度为 2；则保留，不再拆为 1+1。
> + 最差：1。若最后一段绳子长度为1；则应把一份 3+1 替换成 2+2，因为 2x2 > 3x1。
>
> ```C++
> class Solution {
> public:
>     int cuttingRope(int n) {
>         if (n <= 3) return n - 1;
>         int a = n / 3, b = n % 3;
>         if (b == 0) return pow(3, a);
>         else if (b == 1) return pow(3, (a - 1)) * 4;
>         else return pow(3, a) * 2;
>     }
> };
> ```



## * 14-2 剪绳子2

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m - 1] 。请问 k[0]\*k[1]\*...\*k[m - 1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**示例 1：**

```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
```

**示例 2:**

```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```

> 思路：
>
> 处理大数，还没处理好，直接用的 long long 来避免溢出。
>
> ```C++
> class Solution {
> public:
>     int cuttingRope(int n) {
>         if (n <= 3) return n - 1;
>         int a = n / 3, b = n % 3;
>         if (b == 0) return (pow(3, a)) % 1000000007;
>         else if (b == 1) return (pow(3, (a - 1)) * 4) % 1000000007;
>         else return (pow(3, a) * 2) % 1000000007;
>     }
> 
>     long long pow(int a, int b){
>         long long res = 1;
>         while (b > 0){
>             res = (res * (a % 1000000007)) % 1000000007;
>             b = b - 1;
>         }
>         return res;
>     }
> };
> ```



## 15 二进制中 1 的个数

编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量.）。

**示例 1：**

```
输入：n = 11 (控制台输入 00000000000000000000000000001011)
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
```

**示例 2：**

```
输入：n = 128 (控制台输入 00000000000000000000000010000000)
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
```

> 思路：
>
> 位运算，每次与 1 进行与运算，然后右移一位。
>
> ```C++
> class Solution {
> public:
>     int hammingWeight(uint32_t n) {
>         int res = 0;
>         while(n > 0){
>             res = res + int(n & 1);
>             n = n >> 1;
>         }
>         return res;
>     }
> };
> ```



## * 16 数值的整数次方

实现 pow(*x*, *n*) ，即计算 x 的 n 次幂函数（即，x^n）。不得使用库函数，同时不需要考虑大数问题。

**示例 1：**

```
输入：x = 2.00000, n = 10
输出：1024.00000
```

**示例 2：**

```
输入：x = 2.10000, n = 3
输出：9.26100
```

**示例 3：**

```
输入：x = 2.00000, n = -2
输出：0.25000
解释：2^(-2) = (1/2)^2 = 1/4 = 0.25
```



## * 17 打印从 1 到最大的 n 位数

输入数字 `n`，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

**示例 1:**

```
输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]
```

> 如果不考虑大数溢出问题的话，直接遍历即可。
>
> ```C++
> class Solution {
> public:
>     vector<int> printNumbers(int n) {
>         vector<int> res;
>         for (int i = 1; i < pow(10, n); i++){
>             res.push_back(i);
>         }
>         return res;
>     }
> };
> ```
>
> 如果需要考虑大数溢出问题，则需要考虑使用字符串。
>
> 



## 18 删除链表的节点

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

**注意：**此题对比原题有改动

**示例 1:**

```
输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
```

**示例 2:**

```
输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
```

> 思路：
>
> 主要分为两步：定位节点、修改引用。
>
> + **定位节点**：遍历链表，直到 `head.val == val` 时跳出，即可定位目标节点。
> + **修改引用**：设节点 `cur` 的前驱节点为 `pre`，后继节点为 `cur.next`；则执行 `pre.next = cur.next`，即可实现删除 `cur` 节点。
>
> <img src="https://pic.leetcode-cn.com/1613757478-NBOvjn-Picture1.png" style="zoom: 33%;" />
>
> 算法流程：
>
> 1. 特例处理：应删除头节点 `head` 时，直接返回 `head.next` 即可。
> 2. 初始化：`pre = head`，`cur = head.next`
> 3. 定位节点：当 `cur` 为空 或 `cur` 节点值等于 `val` 时跳出。
>    1. 保存当前节点索引，即 `pre = cur`
>    2. 遍历下一节点，即 `cur = cur.next`
> 4. 删除节点：若 `cur` 指向某节点，则执行 `pre.next = cur.next`；若 `cur` 指向 null，代表链表中不包含值为 val 的节点。
> 5. 返回值：返回链表头部节点 `head` 即可。
>
> ```C++
> class Solution {
> public:
>     ListNode* deleteNode(ListNode* head, int val) {
>         if (head -> val == val) {return head -> next;}
>         ListNode* pre = head;
>         ListNode* cur = head -> next;
>         while(cur != NULL){
>             if (cur -> val == val){
>                 pre -> next = cur -> next;
>             }
>             pre = cur;
>             cur = cur -> next;
>         }
>         return head;
>     }
> };
> ```



## * 20 表示数值的字符串





## 21 调整数组顺序使奇数位于偶数前面

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数在数组的前半部分，所有偶数在数组的后半部分。

**示例：**

```
输入：nums = [1,2,3,4]
输出：[1,3,2,4] 
注：[3,1,2,4] 也是正确的答案之一。
```

> 思路一：
>
> 新建一个结果数组 res，遍历输入数组，把奇数添加到 res 数组的前半部分，偶数添加到 res 数组的 后半部分。
>
> 时间复杂度 O(N)；空间复杂度 O(N)
>
> ```C++
> class Solution {
> public:
> 	vector<int> exchange(vector<int>& nums) {
> 		if (nums.size() <= 1) return nums;
> 		vector<int> res(nums.size(), 0);
> 		int m = 0, n = 0;
> 		for (int i = 0; i < nums.size(); i++){
> 			if(nums[i] % 2 != 0){
> 				res[n] = nums[i];
> 				n++;
> 			} else {
> 				res[nums.size() - 1 - m] = nums[i];
> 				m++;
> 			}
> 		}
> 		return res;
> 	}
> };
> ```
>
> 思路二：
>
> 利用双指针，直接在原数组上修改。找到左半部分的偶数，右半部分的奇数，然后交换两者。
>
> <img src="https://pic.leetcode-cn.com/43e965485da89efa688947bc108232f10b65b5ba5c0dbd6a68227a82c7e451e4-Picture1.png" style="zoom: 33%;" />
>
> 时间复杂度 O(N)；空间复杂度 O(1)
>
> ```C++
> class Solution {
> public:
> 	vector<int> exchange(vector<int>& nums) {
> 		if (nums.size() <= 1) return nums;
> 		int i = 0, j = nums.size() - 1;
> 		while (i < j){
> 			while(i < j && ((nums[i] & 1) == 1)) i++;
> 			while(i < j && ((nums[j] & 1) == 0)) j--;
> 			swap(nums[i], nums[j]);
> 		}
> 		return nums;
> 	}
> };
> ```



## 22 链表中倒数第 k 个节点

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

**示例：**

```
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.
```

> 思路一：
>
> 很直观的一种思路，先遍历一遍看链表长度。然后再遍历到 n - k。
>
> ```C++
> class Solution {
> public:
>     ListNode* getKthFromEnd(ListNode* head, int k) {
>         ListNode* cur = head;
>         int m = 0;
>         while(cur != NULL){
>             cur = cur -> next;
>             m++;
>         }
>         cur = head;
>         int n = 0;
>         while(n < m - k){
>             cur = cur -> next;
>             n++;
>         }
>         return cur;
>     }
> };
> ```
>
> 思路二：
>
> 双指针，复杂度和上面一样。设置两个指针，保持距离为 k。
>
> ```C++
> class Solution {
> public:
>     ListNode* getKthFromEnd(ListNode* head, int k) {
>         ListNode* pre = head;
>         ListNode* cur = head;
>         for(int i = 0; i < k; i++){
>             cur = cur -> next;
>         }
>         while(cur != NULL){
>             pre = pre -> next;
>             cur = cur -> next;
>         }
>         return pre;
>     }
> };
> ```



## 24 反转链表

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

**示例:**

```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

**限制：**

```
0 <= 节点个数 <= 5000
```

> 思路一：
>
> 使用双指针。
>
> 时间复杂度 O(N), 空间复杂度 O(1)
>
> ```C++
> class Solution {
> public:
>     ListNode* reverseList(ListNode* head) {
>         ListNode* pre = NULL;
>         ListNode* cur = head;
>         while(cur != NULL){
>             ListNode* temp = cur -> next;
>             cur -> next = pre;
>             pre = cur;
>             cur = temp;
>         }
>         return pre;
>     }
> };
> ```
>
> 思路二：
>
> 使用递归。
>
> 考虑使用递归法遍历链表，当越过尾节点后终止递归，在回溯时修改各节点的 `next` 引用指向。
>
> `recur(cur, pre)  `递归函数：
>
> 1. 终止条件：当 cur 为空，则返回尾节点 pre （即反转链表的头节点）；
> 2. 递归后继节点，记录返回值（即反转链表的头节点）为 res ；
> 3. 修改当前节点 cur 引用指向前驱节点 pre ；
> 4. 返回反转链表的头节点 res ；
>
> `reverseList(head)`  函数：
>
> 调用并返回 `recur(null, head)` 。传入 `null` 是因为反转链表后， `head` 节点指向 `null` ；
>
> ```C++
> class Solution {
> public:
>     ListNode* reverseList(ListNode* head) {
>         return recur(NULL, head);
>     }
> 
>     ListNode* recur(ListNode* pre, ListNode* cur){
>         if (cur == NULL) return pre;
>         ListNode* res = recur(cur, cur -> next);
>         cur -> next = pre;
>         return res;
>     }
> };
> ```



## 25 合并两个排序的链表

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

**示例1：**

```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

**限制：**

```
0 <= 链表长度 <= 1000
```

> 思路：
>
> 设置一个伪头节点（需要初始化，要不然会出错），然后用双指针遍历两个链表。
>
> ```C++
> class Solution {
> public:
>     ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
>         if (l1 == nullptr) return l2;
>         if (l2 == nullptr) return l1;
>         ListNode* res = new ListNode(0), *cur = res;
>         while(l1 != nullptr && l2 != nullptr){
>             if (l1 -> val < l2 -> val){
>                 cur -> next = l1;
>                 l1 = l1 -> next;
>             } else {
>                 cur -> next = l2;
>                 l2 = l2 -> next;
>             }
>             cur = cur -> next;
>         }
>         cur -> next =  l1 == nullptr ? l2 : l1;
>     }
> };
> ```



## 26 树的子结构

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

         3
        / \
       4   5
      / \
     1   2

给定的树 B：

```
   4 
  /
 1
```

返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

**示例 1：**

```
输入：A = [1,2,3], B = [3,1]
输出：false
```

**示例 2：**

```
输入：A = [3,4,5,1,2], B = [4,1]
输出：true
```

> 思路：
>
> 嵌套递归，第一个递归遍历 A 树的所有节点，第二个递归同时遍历 A 和 B 树，判断是否结构相同。
>
> <img src="https://pic.leetcode-cn.com/efe061c2ae8ae6c75b4f36bfd766d60471b4bc985cb330cdae5805043b19f440-Picture5.png" style="zoom: 33%;" />
>
> ```C++
> class Solution {
> public:
>     bool isSubStructure(TreeNode* A, TreeNode* B) {
>         if (B == NULL || A == NULL){
>             return false;
>         }
>         return isContain(A, B) || isSubStructure(A -> left, B) || isSubStructure(A -> right, B);
>     }
> 
>     bool isContain(TreeNode* A, TreeNode* B){
>         if (B == NULL) return true;
>         if (A == NULL || A -> val != B -> val) return false;
>         return isContain(A -> left, B -> left) && isContain(A -> right, B -> right);
>     }
> };
> ```



## 27 二叉树的镜像

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：

```
     4
   /   \
  2     7
 / \   / \
1   3 6   9
```

镜像输出：

```
     4
   /   \
  7     2
 / \   / \
9   6 3   1
```

**示例 1：**

```
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

> 思路1 -- 递归：
>
> 考虑两件事情
>
> 1. 结束条件是什么：遇到了空节点，则返回空节点，不用再往下了
> 2. mirrorTree 函数的作用：实现了子树的左右互换，对于每一颗子树都是这样
>
> ```C++
> class Solution {
> public:
>     TreeNode* mirrorTree(TreeNode* root) {
>         if (root == nullptr) {
>             return nullptr;
>         }
>         TreeNode* tmp = root -> right;
>         root -> right = mirrorTree(root -> left);
>         root -> left = mirrorTree(tmp);
>         return root;
>     }
> };
> ```
>
> 
>
> 思路2 -- 辅助栈：
>
> 自上而下交换左右节点。
>
> ```C++
> class Solution {
> public:
>     TreeNode* mirrorTree(TreeNode* root) {
>         if (root == nullptr) return nullptr;
>         stack<TreeNode*> s;
>         s.push(root);
>         while(!s.empty()){
>             TreeNode* node = s.top();
>             s.pop();
>             if (node -> left != nullptr) s.push(node -> left);
>             if (node -> right != nullptr) s.push(node -> right);
>             swap(node -> left, node -> right); 
>         }
>         return root;
>     }
> };
> ```



## 28 对称的二叉树

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```

但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

```
    1
   / \
  2   2
   \   \
   3    3
```

**示例 1：**

```
输入：root = [1,2,2,3,4,4,3]
输出：true
```

**示例 2：**

```
输入：root = [1,2,2,null,3,null,3]
输出：false
```

> 思路：递归
>
> 对称二叉树的定义：对于树中任意两个对称节点 L 和 R，一定有：
>
> + L.val = R.val
> + L.left.val = R.right.val
> + L.right.val = R.left.val
>
> ```C++
> class Solution {
> public:
>     bool isSymmetric(TreeNode* root) {
>         if (root == nullptr) return true;
>         return recur(root -> left, root -> right);
>     }
> 
>     bool recur(TreeNode* L, TreeNode* R){
>         if (L == nullptr && R == nullptr) return true;
>         if (L == nullptr || R == nullptr || L -> val != R -> val) return false;
>         return recur(L -> left, R -> right) && recur(L -> right, R -> left);
>     }
> };
> ```



## 29 顺时针打印矩阵

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

**示例 1：**

```
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

**示例 2：**

```
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

> 思路：
>
> <img src="https://pic.leetcode-cn.com/ad10b8cab62fdab0261302be2b75f4faceded16a278ddd85281687ab7a6be63e-Picture5.png" style="zoom:33%;" />
>
> ```C++
> class Solution {
> public:
>     vector<int> spiralOrder(vector<vector<int>>& matrix) {
>         vector<int> res;
>         if (matrix.size() == 0 || matrix[0].size() == 0) return res;
>         int t = 0, b = matrix.size() - 1, l = 0, r = matrix[0].size() - 1;
>         while(true){
>             for (int i = l; i <= r; i++){
>                 res.push_back(matrix[t][i]);
>             }
>             if (++t > b) break;
>             for (int i = t; i <= b; i++){
>                 res.push_back(matrix[i][r]);
>             }
>             if (--r < l) break;
>             for (int i = r; i >= l; i--){
>                 res.push_back(matrix[b][i]);
>             }
>             if (--b < t) break;
>             for (int i = b; i >= t; i--){
>                 res.push_back(matrix[i][l]);
>             }
>             if (++l > r) break;
>         }
>         return res;
>     }
> };
> ```



## 30 包含 min 函数的栈

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

**示例:**

```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```

**提示：**

1. 各函数的调用总次数不超过 20000 次

> 思路：辅助栈
>
> <img src="https://pic.leetcode-cn.com/f31f4b7f5e91d46ea610b6685c593e12bf798a9b8336b0560b6b520956dd5272-Picture1.png" style="zoom:48%;" />
>
> ```C++
> class MinStack {
> public:
>     stack<int> A;
>     stack<int> B;
>     /** initialize your data structure here. */
>     MinStack() {
>         
>     }
>     
>     void push(int x) {
>         A.push(x);
>         if (B.size() ==  0 || B.top() >= x){
>             B.push(x);
>         }
>     }
>     
>     void pop() {
>         if (A.top() == B.top()){
>             A.pop();
>             B.pop();
>         } else {
>             A.pop();
>         }
>     }
>     
>     int top() {
>         return A.top();
>     }
>     
>     int min() {
>         return B.top();
>     }
> };
> ```



## 31 栈的压入、弹出序列

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

**示例 1：**

```
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
```

**示例 2：**

```
输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
```

提示：

```
0 <= pushed.length == popped.length <= 1000
0 <= pushed[i], popped[i] < 1000
pushed 是 popped 的排列。
```

> 思路：
>
> <img src="https://pic.leetcode-cn.com/c880f045c03a8e03b7908b2d49b658a9a32ba8f5d40cb19da62db32c7eb58830-Picture1.png" style="zoom: 33%;" />
>
> ```C++
> class Solution {
> public:
>     bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
>         stack<int> s;
>         int j = 0;
>         for (int i = 0; i < pushed.size(); i++){
>             s.push(pushed[i]);
>             while (s.size() > 0 && s.top() == popped[j]){
>                 s.pop();
>                 j++;
>             }
>         }
>         return s.empty();
>     }
> };
> ```



## 32-1 从上到下打印二叉树

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

例如:
给定二叉树: `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回：

```
[3,9,20,15,7]
```

> 思路：BFS / 队列
>
> 按层打印。
>
> ```C++
> class Solution {
> public:
>     vector<int> levelOrder(TreeNode* root) {
>         queue<TreeNode*> q;
>         vector<int> res;
>         if (root == nullptr) return res;
>         q.push(root);
>         while(!q.empty()){
>             res.push_back(q.front() -> val);
>             if(q.front() -> left != nullptr){
>                 q.push(q.front() -> left);
>             }
>             if(q.front() -> right != nullptr){
>                 q.push(q.front() -> right);
>             }
>             q.pop();
>         }
>         return res;
>     }
> };
> ```



## 32-2 从上到下打印二叉树

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

例如:
给定二叉树: `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回其层次遍历结果：

```
[
  [3],
  [9,20],
  [15,7]
]
```

> 思路：BFS
>
> 使用 tmp 数组，然后通过队列中现在的长度来判断这一层的输出个数！
>
> ```C++
> class Solution {
> public:
>     vector<vector<int>> levelOrder(TreeNode* root) {
>         vector<vector<int>> res;
>         queue<TreeNode*> q;
>         if (root == nullptr) return res;
>         q.push(root);
>         while(!q.empty()){
>             vector<int> tmp;
>             for(int i = q.size(); i > 0; i--){
>                 tmp.push_back(q.front() -> val);
>                 if (q.front() -> left != nullptr){
>                     q.push(q.front() -> left);
>                 }
>                 if (q.front() -> right != nullptr){
>                     q.push(q.front() -> right);
>                 }
>                 q.pop();
>             }
>             res.push_back(tmp);
>         }
>         return res;
>     }
> };
> ```



## 32-3 从上到下打印二叉树

请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

例如:
给定二叉树: `[3,9,20,null,null,15,7]`,

```
    3
   / \
  9  20
    /  \
   15   7
```

返回其层次遍历结果：

```
[
  [3],
  [20,9],
  [15,7]
]
```

> 思路：
>
> 判断奇偶反转。
>
> ```C++
> class Solution {
> public:
>     vector<vector<int>> levelOrder(TreeNode* root) {
>         queue<TreeNode*> q;
>         vector<vector<int>> res;
>         if (root == nullptr) return res;
>         q.push(root);
>         while (!q.empty()){
>             vector<int> tmp;
>             for (int i = q.size(); i > 0; i--){
>                 tmp.push_back(q.front() -> val);
>                 if (q.front() -> left != nullptr) q.push(q.front() -> left);
>                 if (q.front() -> right != nullptr) q.push(q.front() -> right);
>                 q.pop();
>             }
>             if (res.size() %2 == 1) reverse(tmp.begin(), tmp.end());
>             res.push_back(tmp);
>         }
>         return res;
>     }
> };
> ```



## 33 二叉树的后续遍历序列

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 `true`，否则返回 `false`。假设输入的数组的任意两个数字都互不相同。

参考以下这颗二叉搜索树：

```
     5
    / \
   2   6
  / \
 1   3
```

**示例 1：**

```
输入: [1,6,3,2,5]
输出: false
```

**示例 2：**

```
输入: [1,3,2,6,5]
输出: true
```

> 思路：递归分治
>
> **二叉搜索树定义：** 左子树中所有节点的值 < 根节点的值；右子树中所有节点的值 > 根节点的值；其左、右子树也分别为二叉搜索树。
>
> 根据二叉搜索树的定义，可以通过递归，判断所有子树的 **正确性** （即其后序遍历是否满足二叉搜索树的定义） ，若所有子树都正确，则此序列为二叉搜索树的后序遍历。
>
> <img src="https://pic.leetcode-cn.com/4a2780853b72a0553194773ff65c8c81ddcc4ee5d818cb3528d5f8dd5fa3b6d8-Picture1.png" style="zoom:33%;" />
>
> + 终止条件：当 `i >= j`，说明此树节点数量  <= 1，无需判别正确性，直接返回 true。
> + 返回值：所有子树都需正确才可判定正确，因此使用 **与逻辑符** &&&& 连接。
>   + `p == j` : 判断 **此树** 是否正确。
>   + `recur(i, m - 1)` :  判断 **此树的左子树** 是否正确。
>   + `recur(m, j - 1)` : 判断 **此树的右子树** 是否正确。
>
> ```C++
> class Solution {
> public:
>     bool verifyPostorder(vector<int>& postorder) {
>         return recur(postorder, 0, postorder.size() - 1);
>     }
> 
>     bool recur(vector<int>& postorder, int i, int j){
>         if (i >= j) return true;
>         int p = i;
>         while (postorder[p] < postorder[j]) p++;
>         int m = p;
>         while (postorder[p] > postorder[j]) p++;
>         return p == j && recur(postorder, i, m - 1) && recur(postorder, m, j - 1);
>     }
> };
> ```



## 34 二叉树中和为某一值的路径

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

**示例 1：**

<img src="https://assets.leetcode.com/uploads/2021/01/18/pathsumii1.jpg" style="zoom:50%;" />

```
输入：root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
输出：[[5,4,11,2],[5,8,4,5]]
```

**示例 2：**

<img src="https://assets.leetcode.com/uploads/2021/01/18/pathsum2.jpg" style="zoom: 50%;" />

```
输入：root = [1,2,3], targetSum = 5
输出：[]
```

**示例 3：**

```
输入：root = [1,2], targetSum = 0
输出：[]
```

> 思路：递归，回溯法
>
> ```C++
> class Solution {
> public:
>     vector<vector<int>> res;
>     vector<int> path;
>     vector<vector<int>> pathSum(TreeNode* root, int target) {
>         recur(root, target);
>         return res;
>     }
> 
>     void recur(TreeNode* root, int target){
>         if (root == nullptr) return;
>         path.push_back(root -> val);
>         target = target - root -> val;
>         if (target == 0 && root -> left == nullptr && root -> right == nullptr){
>             res.push_back(path);
>         }
>         recur(root -> left, target);
>         recur(root -> right, target);
>         path.pop_back();
>     }
> };
> ```



## 35 复杂链表的复制

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

**示例 1：**

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e1.png" style="zoom: 33%;" />

```
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```

**示例 2：**

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e2.png" style="zoom:33%;" />

```
输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
```

**示例 3：**

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/01/09/e3.png" style="zoom:33%;" />

```
输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
```

**示例 4：**

```
输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。
```

> 思路：哈希表
>
> 利用哈希表的查询特点，考虑构建 **原链表节点** 和 **新链表对应节点** 的键值对映射关系，再遍历构建新链表各节点的 `next` 和 `random` 引用指向即可。
>
> ```C++
> class Solution {
> public:
>     Node* copyRandomList(Node* head) {
>         unordered_map<Node*, Node*> m;
>         Node* cur = head;
>         while(cur != nullptr){
>             m[cur] = new Node(cur -> val);
>             cur = cur -> next;
>         }
>         cur = head;
>         while(cur != nullptr){
>             m[cur] -> next = m[cur -> next];
>             m[cur] -> random = m[cur -> random];
>             cur = cur -> next;
>         }
>         return m[head];
>     }
> };
> ```



## 36 二叉搜索树与双向链表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

为了让您更好地理解问题，以下面的二叉搜索树为例：

<img src="https://assets.leetcode.com/uploads/2018/10/12/bstdlloriginalbst.png" style="zoom: 50%;" />

我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

下图展示了上面的二叉搜索树转化成的链表。“head” 表示指向链表中有最小元素的节点。

<img src="https://assets.leetcode.com/uploads/2018/10/12/bstdllreturndll.png" style="zoom: 50%;" />

特别地，我们希望可以就地完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中的第一个节点的指针。

> 思路：中序遍历
>
> ```C++
> // 打印中序遍历
> void dfs(Node* root) {
>     if(root == nullptr) return;
>     dfs(root->left); // 左
>     cout << root->val << endl; // 根
>     dfs(root->right); // 右
> }
> ```
>
> <img src="https://pic.leetcode-cn.com/1599401091-PKIjds-Picture1.png" style="zoom: 33%;" />
>
> ```C++
> class Solution {
> public:
>     Node* treeToDoublyList(Node* root) {
>         if (root == nullptr) return nullptr;
>         head = dfs(root);
>         head -> left = pre;
>         pre -> right = head;
>         return head;
> 
>     }
> 
> private:
>     Node* pre, *head;
>     Node* dfs(Node* cur){
>         if (cur == nullptr) return nullptr;
>         dfs(cur -> left);
>         if (pre == nullptr) head = cur;
>         else {
>             pre -> right = cur;
>             cur -> left = pre;
>         }
>         pre = cur;
>         dfs(cur -> right);
>         return head;
>     }
> };
> ```



## 37 序列化二叉树

请实现两个函数，分别用来序列化和反序列化二叉树。

你需要设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

**示例：**

<img src="https://assets.leetcode.com/uploads/2020/09/15/serdeser.jpg" style="zoom:50%;" />

```
输入：root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]
```

> 思路： 层序遍历
>
> ```C++
> class Codec {
> public:
> 
>     // Encodes a tree to a single string.
>     string serialize(TreeNode* root) {
>         queue<TreeNode*> q;
>         if (root == nullptr) return "[]";
>         q.push(root);
>         string res = "[";
>         while(!q.empty()){
>             if (q.front() == nullptr) {
>                 res = res + "null,";
>             } else {
>                 res = res + to_string(q.front() -> val) + ",";
>                 q.push(q.front() -> left);
>                 q.push(q.front() -> right);
>             }
>             q.pop();
>         }
>         res = res + "]";
>         return res;
>     }
> 
>     // Decodes your encoded data to tree.
>     TreeNode* deserialize(string data) {
>         if (data.compare("[]") == 0) return nullptr;
>         vector<TreeNode*> v = split(data);
>         queue<TreeNode*> q;
>         q.push(v[0]);
>         int i = 1;
>         while(!q.empty()){
>             q.front() -> left = v[i];
>             if (v[i] != nullptr) q.push(v[i]);
>             i++;
>             q.front() -> right = v[i];
>             if (v[i] != nullptr) q.push(v[i]);
>             i++;
>             q.pop();
>         }
>         return v[0];
>     }
> 
> 
>     vector<TreeNode*> split(string s){
>         vector<TreeNode*> v;
>         string tmp = "";
>         for (int i = 1; i < s.size() - 1; i++){
>             if(s[i] != ','){
>                 tmp += s[i];
>             } else {
>                 if (tmp.compare("null") == 0){
>                     v.push_back(nullptr);
>                 } else{
>                     TreeNode* node = new TreeNode(stoi(tmp));
>                     v.push_back(node);
>                 }
>                 tmp = "";
>             }
>         }
>         return v;
>     }
> };
> ```



## 38 字符串的全排列

输入一个字符串，打印出该字符串中字符的所有排列。

**示例1:**

```
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
```

**示例2:**

```
输入：s = "aab"
输出：["aab","aba","baa"]
```

> 思路：
>
> 回溯法，主要是怎么有效处理重复字符的情况
>
> ```C++
> class Solution {
> public:
>     vector<string> res;
>     vector<bool> used;
>     vector<string> permutation(string s) {
>         sort(s.begin(), s.end());
>         used.assign(s.size(), false);
>         string path = "";
>         backtrack(path, s);
>         return res;
>     }
> 
>     void backtrack(string path, string s){
>         if (path.size() == s.size()){
>             res.push_back(path);
>             return;
>         }
>         for (int i = 0; i < s.size(); i++){
>             if (isValid(i, s)){
>                 path.push_back(s[i]);
>                 used[i] = true;
>                 backtrack(path, s);
>                 path.pop_back();
>                 used[i] = false;
>             }
>         }
>     }
>     
>     bool isValid(int i, string s){
>         if (!used[i]){
>             if(i == 0) return true;
>             if (s[i] == s[i-1] && !used[i-1]){
>                 return false;
>             } else {
>                 return true;
>             }
>         } else {
>             return false;
>         }
>     }
> };
> ```



## 39 数组中出现次数超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

**示例 1:**

```
输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
输出: 2
```

> 思路1: HashMap
>
> ```C++
> class Solution {
> public:
>     int majorityElement(vector<int>& nums) {
>         map<int, int>m;
>         for (int i = 0; i < nums.size(); i++){
>             if (!m[nums[i]]){
>                 m[nums[i]] = 1;
>             } else {
>                 m[nums[i]] += 1;
>             }
>         }
>         map<int, int>::iterator iter = m.begin();
>         while (iter != m.end()){
>             if (iter -> second > nums.size() / 2){
>                 return iter -> first;
>             }
>             iter++;
>         }
>         return -1;
>     }
> };
> ```
>
> 思路2: 排序
>
> ```C++
> class Solution {
> public:
>     int majorityElement(vector<int>& nums) {
>         sort(nums.begin(), nums.end());
>         return nums[nums.size() / 2];
>     }
> };
> ```
>
> 思路3: 摩尔投票法（同归于尽法）
>
> <img src="https://pic.leetcode-cn.com/1603612327-bOQxzq-Picture1.png" style="zoom: 50%;" />
>
> ```C++
> class Solution {
> public:
>     int majorityElement(vector<int>& nums) {
>         int votes = 0;
>         int x;
>         for (int i = 0; i < nums.size(); i++){
>             if (votes == 0){
>                 x = nums[i];
>             }
>             if (nums[i] == x){
>                 votes++;
>             } else {
>                 votes--;
>             }
>         }
>         return x;
>     }
> };
> ```







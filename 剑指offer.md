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
vector<int> res (arr.begin(), arr.begin() + k);
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

> 思路：双指针
```C++
class Solution {
public:
    string replaceSpace(string s) {
        int cnt = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == ' ') cnt++;
        }
        int originSize = s.size();
        s.resize(originSize + cnt * 2);
        int i = originSize - 1, j = s.size() - 1;
        while (i < j) {
            if (s[i] == ' ') {
                s[j] = '0';
                s[j - 1] = '2';
                s[j - 2] = '%';
                j -= 3;
                i--;
            } else {
                s[j] = s[i];
                i--;
                j--;
            }
        }
        return s;
    }
};
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
> /**
>  * Definition for a binary tree node.
>  * struct TreeNode {
>  *     int val;
>  *     TreeNode *left;
>  *     TreeNode *right;
>  *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
>  * };
>  */
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
>                 res += "null,";
>             } else {
>                 res += to_string(q.front() -> val) + ",";
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



> 递归解法：理论上效率稍差，但是思路清晰
>
> ```java
> public class Codec {
> 
>     final String NULL = "#";
>     final String SEPARATOR = ",";
>     public String serialize(TreeNode node) {
>         StringBuilder sb = new StringBuilder();
>         serialize(node, sb);
>         return sb.toString();
>     }
> 
>     public void serialize(TreeNode node, StringBuilder builder){
>         if(node == null) {
>             builder.append(NULL).append(SEPARATOR);
>             return;
>         }
> 
>         serialize(node.left, builder);
>         serialize(node.right, builder);
> 
>         builder.append(node.val).append(SEPARATOR);
> 
>     }
> 
>     public TreeNode deserialize(String data) {
>         LinkedList<String> nodes = new LinkedList<String>();
>         for(String val : data.split(SEPARATOR)) {
>             nodes.add(val);
>         }
>         return deserialize(nodes);
> 
>     }
> 
>     public TreeNode deserialize(LinkedList<String> nodes) {
>         if(nodes.isEmpty()) return null;
>         
>         String node = nodes.removeLast();
>         if(node.equals(NULL)) return null;
>         TreeNode root = new TreeNode(Integer.parseInt(node));
>         
>         root.right = deserialize(nodes);
>         root.left = deserialize(nodes);
> 
>         return root;
> 
> 
>     }
> }
> ```
>
> 

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



## 40 最小的 k 个数

输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

**示例 1：**

```
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```

**示例 2：**

```
输入：arr = [0,1,2,1], k = 1
输出：[0]
```

> 思路：
>
> 哨兵划分操作
>
> ```C++
> class Solution {
> public:
>     vector<int> getLeastNumbers(vector<int>& arr, int k) {
>         if (arr.size() == 0) return arr;
>         quicksort(arr, 0, arr.size() - 1);
>         vector<int> res (arr.begin(), arr.begin() + k);
>         return res;
> 
>     }
> 
>     void quicksort(vector<int>& a, int l, int r){
>         if (l >= r) return ;
>         int i = l, j = r;
>         while(i < j) {
>             while (i < j && a[j] >= a[l]) j--;
>             while (i < j && a[i] <= a[l]) i++;
>             swap(a[i], a[j]);
>         }
>         swap(a[l], a[i]);
>         quicksort(a, l, i-1);
>         quicksort(a, i+1, r);
>     }
> };
> ```
>
> 根据 i 和 k 的大小，减去不必要的操作。
>
> ```C++
> class Solution {
> public:
>     int k;
>     vector<int> getLeastNumbers(vector<int>& arr, int k) {
>         if (arr.size() == 0) return arr;
>         this -> k = k;
>         quicksort(arr, 0 ,arr.size() - 1);
>         vector<int> res(arr.begin(), arr.begin() + k);
>         return res;
>     }
> 
>     void quicksort(vector<int>& arr, int l, int r){
>         if (l >= r) return;
>         int i = l, j = r;
>         while(i < j){
>             while(i < j && arr[j] >= arr[l]) j--;
>             while(i < j && arr[i] <= arr[l]) i++;
>             swap(arr[i], arr[j]); 
>         }
>         swap(arr[l], arr[i]);
>         if (i <= k) quicksort(arr, i + 1, r);  // 说明左边的i个数都属于最小的前k个数，那么只用对右半部分排序。
>         if (i > k) quicksort(arr, l, i - 1);
>     }
> };
> ```



## 42 连续子数组的最大和

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

**示例1:**

```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

> 思路：
>
> 如果前面的和小于0，则令当前最大值为当前遍历的值。
>
> ```C++
> class Solution {
> public:
> 	int maxSubArray(vector<int>& nums) {
> 		int res = nums[0];
> 		int cur = nums[0];
> 		for (int i = 1; i < nums.size(); i++){
> 			if (cur < 0) cur = nums[i];
> 			else cur = cur + nums[i];
> 			res = max(res, cur);
> 		}
> 		return res;
> 	}
> };
> ```
>
> 动态规划，`nums[i]` 表示到 `i` 位置连续子数组的最大和。
>
> ```C++
> class Solution {
> public:
> 	int maxSubArray(vector<int>& nums) {
> 		for (int i = 1; i < nums.size(); i++){
>             if (nums[i - 1] > 0){
>                 nums[i] = nums[i] + nums[i - 1];
>             }
>         }
>         return *max_element(nums.begin(), nums.end());
> 	}
> };
> ```



## 43 1-n 整数中 1 出现的次数

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

**示例 1：**

```
输入：n = 12
输出：5
```

**示例 2：**

```
输入：n = 13
输出：6
```

> 思路1: 暴力求解
>
> ```C++
> class Solution {
> public:
> 	int countDigitOne(int n) {
> 		int res = 0;
> 		for (int i = 0; i <= n; i++){
> 			int tmp = i;
> 			while(tmp > 0){
> 				if (tmp % 10 == 1){
> 					res++;
> 				}
> 				tmp /= 10;
> 			}
> 		}
> 		return res;
> 	}
> };
> ```
>
> 思路2：找规律
>
> <img src="https://pic.leetcode-cn.com/1f7e8ce0bf03c7fc974082c32ec909ebffc6429636ec46cecd492604c65ec87f-Picture6.png" style="zoom:48%;" />
>
> ```C++
> class Solution {
> public:
> 	int countDigitOne(int n) {
> 		int res = 0;
> 		long digit = 1;
> 		int cur  = n % 10;
> 		int high = n / 10;
> 		int low = 0;
> 		while(high != 0 || cur != 0){
> 			if (cur == 0) res += high * digit;
> 			else if (cur == 1) res += high * digit + low + 1;
> 			else res += (high + 1) * digit;
> 			low += cur * digit;
> 			cur = high % 10;
> 			high /= 10;
> 			digit *= 10;
> 		}
> 		return res;
> 	}
> };
> ```



## 45 把数组排成最小的数

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

**示例 1:**

```
输入: [10,2]
输出: "102"
```

**示例 2:**

```
输入: [3,30,34,5,9]
输出: "3033459"
```

> 思路：
>
> <img src="https://pic.leetcode-cn.com/95e81dbccc44f26292d88c509afd68204a86b37d342f83d109fa7aa0cd4a6049-Picture1.png" style="zoom:48%;" />
>
> ```C++
> class Solution {
> public:
>     string minNumber(vector<int>& nums) {
>         vector<string> strs;
>         string res;
>         for(int i = 0; i < nums.size(); i++){
>             strs.push_back(to_string(nums[i]));
>         }
>         sort(strs.begin(), strs.end(), [](string& x, string& y){ return x + y < y + x; });
>         for(int i = 0; i < strs.size(); i++){
>             res.append(strs[i]);
>         }
>         return res;
>     }
> };
> ```



## 46 把数字翻译成字符串



给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

**示例 1:**

```
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
```

> 思路1：动态规划（有条件的斐波那契数列）
>
> ```C++
> class Solution {
> public:
> 	int translateNum(int num) {
> 		vector<int> nums;
> 		int pre_2 = 1;
> 		int pre_1 = 1;
> 		int res = 1;
> 		int n = num;
> 		while(n > 0){
> 			nums.insert(nums.begin(), n % 10);
> 			n /= 10;
> 		}
> 		for (int i = 1; i < nums.size(); i++){
> 			if ((nums[i-1] * 10 + nums[i] <= 25) && nums[i-1] != 0){
> 				res = pre_1 + pre_2;
> 			} else {
> 				res = pre_1;
> 			}
> 			pre_2 = pre_1;
> 			pre_1 = res;
> 		}
> 		return res;
> 	}
> };
> ```



## 47 礼物的最大价值

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

**示例 1:**

```
输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```

> 思路：动态规划
>
> 一、处理边界问题
>
> 时间复杂度 O(MN); 空间复杂度 O(1)
>
> ```C++
> class Solution {
> public:
>     int maxValue(vector<vector<int>>& grid) {
>         for (int i = 0; i < grid.size(); i++){
>             for (int j = 0; j < grid[0].size(); j++){
>                 if (i == 0 && j == 0) continue;
>                 else if (i == 0) grid[0][j] += grid[0][j-1];
>                 else if (j == 0) grid[i][0] += grid[i-1][0];
>                 else {
>                     grid[i][j] = max(grid[i-1][j], grid[i][j-1]) + grid[i][j];
>                 }
>             }
>         }
>         return grid[grid.size() - 1][grid[0].size() - 1];
>     }
> };
> ```
>
> 二、添加辅助边界
>
> 时间复杂度 O(MN); 空间复杂度 O(MN)
>
> ```C++
> class Solution {
> public:
>     int maxValue(vector<vector<int>>& grid) {
>         vector<vector<int>> dp (grid.size() + 1, vector<int>(grid[0].size() + 1, 0));
>         for (int i = 1; i < dp.size(); i++){
>             for (int j = 1; j < dp[0].size(); j++){
>                 dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i-1][j-1];
>             }
>         }
>         return dp[dp.size()-1][dp[0].size()-1];
>     }
> };
> ```



## 48 最长不含重复字符的子字符串

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

**示例 1:**

```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2:**

```
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

**示例 3:**

```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

> 思路1：动态规划 + 哈希表
>
> 动态规划：
>
> + 状态定义：设动态规划列表 `dp` , `dp[j]` 代表以字符 `s[j]` 为结尾的“最长不重复子字符串”的长度。
> + 转移方程：固定右边界 `j`，设字符 `s[j]` 左边距离最近的相同字符为 `s[i]` ，即 `s[i]=s[j]`
>   + 当 `i < 0` , `s[j]` 左侧没有相同字符，则 `dp[j] = dp[j-1] + 1`
>   + 当 `dp[j-1] < j-i` , 说明字符 `s[j]` 在子字符串 `dp[j-1]` 区间之外，`dp[j] = dp[j-1] + 1`
>   + 当 `dp[j-1] >= j-i` , 说明字符 `s[j]` 在子字符串 `dp[j-1]` 区间之中，则 `dp[j] = j - i` 
>
> ```C++
> class Solution {
> public:
>     int lengthOfLongestSubstring(string s) {
>         if (s == "") return 0;
>         map<int,int> dic;
>         vector<int> dp (s.size(), 0);
>         dp[0] = 1;
>         dic[s[0]] = 0;
>         for (int i = 1; i < s.size(); i++){
>             if (dic.find(s[i]) != dic.end()){
>                 int pre = dic[s[i]];
>                 if (dp[i-1]  < i - pre){
>                     dp[i] = dp[i-1] + 1;
>                 } else {
>                     dp[i] = i - pre;
>                 }
>             } else {
>                 dp[i] = dp[i-1] + 1;
>             }
>             dic[s[i]] = i;
>         }
>         return *max_element(dp.begin(), dp.end());
>     }
> };
> ```





## 49 丑数

我们把只包含质因子 2、3、5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

示例：

```
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```

> 思路：
>
> 设置3个索引a, b, c，分别记录前几个数已经被乘2， 乘3， 乘5了，比如a表示前(a-1)个数都已经乘过一次2了，下次应该乘2的是第a个数；b表示前(b-1)个数都已经乘过一次3了，下次应该乘3的是第b个数；c表示前(c-1)个数都已经乘过一次5了，下次应该乘5的是第c个数；
>
> ```C++
> class Solution {
> public:
>     int nthUglyNumber(int n) {
>         vector<int> dp(n, 0);
>         dp[0] = 1;
>         int a = 0, b = 0, c = 0;
>         for (int i = 1; i < n; i++){
>             int n1 = dp[a] * 2;
>             int n2 = dp[b] * 3;
>             int n3 = dp[c] * 5;
>             dp[i] = min(min(n1, n2), n3);
>             if (dp[i] == n1) a++;
>             if (dp[i] == n2) b++;
>             if (dp[i] == n3) c++;
>         }
>         return dp[n-1];
>     }
> };
> ```





## 50 第一个只出现一次的字符

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

**示例 1:**

```
输入：s = "abaccdeff"
输出：'b'
```

**示例 2:**

```
输入：s = "" 
输出：' '
```

> 思路：哈希表
>
> 第二次循环遍历字符串
>
> ```C++
> class Solution {
> public:
>     char firstUniqChar(string s) {
>         map<char, bool> dic;
>         for(int i = 0; i < s.size(); i++){
>             if (dic.find(s[i]) == dic.end()){
>                 dic[s[i]] = true;
>             } else if (dic[s[i]] == true) {
>                 dic[s[i]] = false;
>             }
>         }
>         for (char c : s){
>             if (dic[c]) return c;
>         }
>         return ' ';
>     }
> };
> ```
>
> 利用辅助 vector 构建有序 map 遍历，在字符串很长的时候，能够减少循环次数。
>
> ```C++
> class Solution {
> public:
>     char firstUniqChar(string s) {
>         map<char, bool> dic;
>         vector<char> keys;
>         for(int i = 0; i < s.size(); i++){
>             if (dic.find(s[i]) == dic.end()){
>                 dic[s[i]] = true;
>                 keys.push_back(s[i]);
>             } else if (dic[s[i]] == true) {
>                 dic[s[i]] = false;
>             }
>         }
>         for (char c : keys){
>             if (dic[c]) return c;
>         }
>         return ' ';
>     }
> };
> ```





## 52 两个链表的第一个公共节点

输入两个链表，找出它们的第一个公共节点。

如下面的两个链表**：**

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)

在节点 c1 开始相交。

> 思路：
> 链表A ：(a-c) + c
>
> 链表B ：(b-c) + c
>
> (a-c) + c + (b-c) = (b-c) + c + (a-c)
>
> ```C++
> class Solution {
> public:
>     ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
>         ListNode* A = headA;
>         ListNode* B = headB;
>         while (A != B){
>             if (A != nullptr) {
>                 A = A -> next;
>             } else {
>                 A = headB;
>             }
>             if (B != nullptr) {
>                 B = B -> next;
>             } else {
>                 B = headA;
>             }
>         }
>         return A;
>     }
> };
> ```



## 53-I 在排序数组中查找数字 I

统计一个数字在排序数组中出现的次数。

**示例 1:**

```
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
```

**示例 2:**

```
输入: nums = [5,7,7,8,8,10], target = 6
输出: 0
```

> 思路：二分法
>
> <img src="https://pic.leetcode-cn.com/b4521d9ba346cad9e382017d1abd1db2304b4521d4f2d839c32d0ecff17a9c0d-Picture1.png" style="zoom:48%;" />
>
> ```C++
> class Solution {
> public:
>     int search(vector<int>& nums, int target) {
>         int i = 0, j = nums.size() - 1;
>         while (i <= j) {
>             int m = (i + j) / 2;
>             if (nums[m] < target) i = m + 1;
>             else j = m - 1;
>         }
>         int left = j;
>         i = 0, j = nums.size() - 1;
>         while (i <= j) {
>             int m = (i + j) / 2;
>             if (nums[m] <= target) i = m + 1;
>             else j = m - 1;
>         }
>         int right = i;        
>         return right - left - 1;
>     }
> };
> ```





## 53-II 0~n-1中缺失的数字

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

**示例 1:**

```
输入: [0,1,3]
输出: 2
```

**示例 2:**

```
输入: [0,1,2,3,4,5,6,7,9]
输出: 8
```

> 思路1: 二分法条件 `while(l < r)`
>
> - 排序数组中的搜索问题，首先想到 **二分法** 解决。
> - 根据题意，数组可以按照以下规则划分为两部分。
>   - 左子数组：`nums[i] = i` 
>   - 右子数组：`nums[i] != i` 
> - 缺失的数字等于 **“右子数组的首位元素”** 对应的索引；因此考虑使用二分法查找 “右子数组的首位元素” 。
>
> ```C++
> class Solution {
> public:
>     int missingNumber(vector<int>& nums) {
>         int l = 0;
>         int r = nums.size() - 1;
>         while (l < r) {
>             int m = (l + r) / 2;
>             if (nums[m] == m) {
>                 l = m + 1;
>             } else {
>                 r = m;
>             }
>         }
>         return nums[r] != r ? r : r + 1;  // 当 l = r 的时候，还可能出现[0,1,2,3]，最终l和r都出现在右边界。
>     }
> };
> ```
>
> 思路2：二分法条件 `while(l <= r)` 
>
> - 缺失的数字等于 **“右子数组的首位元素”** 对应的索引；因此考虑使用二分法查找 “右子数组的首位元素” 。
>
> 循环二分： 当 `l <= r` 时循环，当闭区间 `[i, j]` 为空时跳出循环。
>
> 1. 计算中点 `m` 
> 2. `nums[m] == m` : 则 “右子数组的首位元素” 一定在闭区间 `[m+1, r]` , 令 `l = m+1` 
> 3. `nums[m] != m` : 则 “左子数组的末位元素” 一定在闭区间 `[l, m-1]` , 令 `r = m-1` 
>
> 跳出时，变量 `l` 和 `r` 分别指向 “右子数组的首位元素” 和 “左子数组的末位元素” 。因此返回 `l` 即可。
>
> ```C++
> class Solution {
> public:
>     int missingNumber(vector<int>& nums) {
>         int l = 0;
>         int r = nums.size() - 1;
>         while (l <= r){
>             int m = (l + r) / 2;
>             if (nums[m] == m) {
>                 l = m + 1;
>             } else {
>                 r = m - 1;
>             }
>         }
>         return l;
>     }
> };
> ```



## 54 二叉搜索树的第 k 大节点

给定一棵二叉搜索树，请找出其中第 `k` 大的节点的值。

**示例 1:**

```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
```

**示例 2:**

```
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 4
```

> 思路：中序遍历倒序
>
> ```C++
> class Solution {
> public:
>     int res, k;
>     int kthLargest(TreeNode* root, int k) {
>         this -> k = k;
>         dfs(root);
>         return res;
>     }
> 
>     void dfs(TreeNode* root) {
>         if (root == nullptr || k == 0) return;
>         dfs(root -> right);
>         if (--k == 0) res = root -> val;
>         dfs(root -> left);
>     }
> };
> ```



## 55-I 二叉树的深度

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

例如：

给定二叉树 `[3,9,20,null,null,15,7]`，

```
    3
   / \
  9  20
    /  \
   15   7
```

返回它的最大深度 3 。

> 思路1: 递归 DFS
>
> ```C++
> class Solution {
> public:
>     int maxDepth(TreeNode* root) {
>         if (root == nullptr) return 0;
>         return max(maxDepth(root -> left), maxDepth(root -> right)) + 1;
>     }
> };
> ```
>
> 思路2: 队列 BFS
>
> ```C++
> class Solution {
> public:
>     int maxDepth(TreeNode* root) {
>         if (root == nullptr) return 0;
>         queue<TreeNode*> q;
>         q.push(root);
>         int res = 0;
>         while(!q.empty()){
>             int n = q.size();
>             for (int i = 0; i < n; i++){
>                 TreeNode* node = q.front();
>                 if (node -> left != nullptr) q.push(node -> left);
>                 if (node -> right != nullptr) q.push(node -> right);
>                 q.pop();
>             }
>             res++;
>         }
>         return res;
>     }
> };
> ```


## 55-II 平衡二叉树

输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

**示例 1:**

给定二叉树 `[3,9,20,null,null,15,7]`

```
    3
   / \
  9  20
    /  \
   15   7
```

返回 `true` 。

**示例 2:**

给定二叉树 `[1,2,2,3,3,null,null,4,4]`

```
       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
```

返回 `false` 。

> 思路1: 先序遍历 + 判断深度 （从顶至底）

```C++
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        if (root == nullptr) return true;
        return abs(depth(root -> left) - depth(root -> right)) <= 1 && isBalanced(root -> left) && isBalanced(root -> right);
    }

    int depth(TreeNode* root){
        if (root == nullptr) return 0;
        return max(depth(root -> left), depth(root -> right)) + 1;
    }

};
```

## 56-I 数组中数字出现的次数
一个整型数组 `nums` 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。  
**示例 1：**
```
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
```
**示例 2：**
```
输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]
```
> 思路：位运算(`&`, 优先级很低，低于`==`)
```C++
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        int x = 0, y = 0, m = 1;
        int s = 0;
        for (int i = 0; i < nums.size(); i++){
            s ^= nums[i];
        }
        while ((s & 1) == 0){
            s >>= 1;
            m <<= 1;
        }
        for (int i = 0; i < nums.size(); i++){
            if ((nums[i] & m) == 0){
                x ^= nums[i];
            } else {
                y ^= nums[i];
            }
        }
        return vector<int> {x, y};
    }
};
```

## 56-II 数组中数字出现的次数 
在一个数组 `nums` 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。  
**示例 1：**
```
输入：nums = [3,4,3,3]
输出：4
```
**示例 2：**
```
输入：nums = [9,1,7,9,7,9,7]
输出：1
```
限制：
```
1 <= nums.length <= 10000
1 <= nums[i] < 2^31
```

> 思路：判断每个比特位总数是不是 3 的倍数
```C++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        vector<int> bits (32, 0);
        int res = 0;
        for (int i = 0; i < 32; i++){
            for (int j = 0; j < nums.size(); j++){
                bits[i] += ((nums[j] >> i) & 1);
            }
        }
        for (int i = 0; i < bits.size(); i++){
            if (bits[i] % 3 != 0){
                res = res + (1 << i);
            }
        }
        return res;
    }
};
```

## 57 和为 s 的两个数字
输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

**示例 1：**
```
输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
```
**示例 2：**
```
输入：nums = [10,26,30,31,47,60], target = 40
输出：[10,30] 或者 [30,10]
```
> 思路：双指针
```C++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int a = 0;
        int b = nums.size() - 1;
        while (nums[a] + nums[b] != target) {
            if (nums[a] + nums[b] > target) {
                b--;
            } else if (nums[a] + nums[b] < target) {
                a++;
            }
        }
        return vector<int> {nums[a], nums[b]};
    }
};
```

## 57-II 和为 s 的连续正数序列
输入一个正整数 `target` ，输出所有和为 `target` 的连续正整数序列（至少含有两个数）。
序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

**示例 1：**
```
输入：target = 9
输出：[[2,3,4],[4,5]]
```

**示例 2：**
```
输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
```

> 思路：双重循环，会有一些重复的相加操作
```C++
class Solution {
public:
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> res;
        for (int i = 1; i < (target/2 + 1); i++){
            int j = i;
            int s = j;
            vector<int> tmp;
            while (s <= target){
                if (s == target){
                    for (int a = i; a <= j; a++){
                        tmp.push_back(a);
                    }
                    res.push_back(tmp);
                } 
                j++;
                s += j;
            }
        }
        return res;
    }
};
```

> 思路：滑动窗口，没有重复操作
```C++
class Solution {
public:
	vector<vector<int>> findContinuousSequence(int target) {
		vector<vector<int>> res;
		int i = 1, j = 2, s = 3;
		while (i < target/2 + 1){
			if (s < target) {
				j++;
				s += j;
			} else if (s > target){
				s -= i;
				i++;
			} else{
				vector<int> tmp;
				for (int m = i; m <= j; m++){
					tmp.push_back(m);
				}
				res.push_back(tmp);
				s -= i;
				i++;
			}
		}
		return res;
	}
};
```

## 58-I 翻转单词顺序
输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。
**示例 1：**
```
输入: "the sky is blue"
输出: "blue is sky the"
```
**示例 2：**
```
输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
```
**示例 3：**
```
输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
```

说明：
+ 无空格字符构成一个单词。
+ 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
+ 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。

> 思路： 双指针
```C++
class Solution {
public:
	string reverseWords(string s) {
		string res;
		int i = s.size() - 1, j = s.size() - 1;
		while (i >= 0){
			if (s[i] != ' ' && (i != 0)){
				i--;
			} else if (s[i] != ' ' && (i == 0)) {
				res += s.substr(0, j - i + 1);
				i--;
			} else {
				if (j != i){
					res += s.substr(i + 1, j - i);
					res += " ";
				}
				i--;
				j = i;
			}
		}
		if (res.back() == ' ') res.pop_back();
		return res;
	}
};
```

## 58-II 左旋转字符串
字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

**示例 1：**
```
输入: s = "abcdefg", k = 2
输出: "cdefgab"
```
**示例 2：**
```
输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"
```

> 思路：字符串切片
```C++
class Solution {
public:
    string reverseLeftWords(string s, int n) {
        string res = s.substr(n, s.size()-1) + s.substr(0, n);
        return res;
    }   
};
```

## 59-I 滑动窗口的最大值
给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。

**示例:**
```
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

> 思路： 双向队列，单调队列
1. deque 内 **仅包含窗口内的元素**，每轮窗口滑动移除了元素 nums[i-1]，需将 deque 内对应的元素一起删除。
2. deque 内的元素 **非严格递减**， 每轮窗口滑动添加了元素 nums[j+1]，需将 deque 内所小于 nums[j+1] 的元素删除。

```C++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        int j = 0;
        int i = 1 - k;
        deque<int> tmp;
        while (j < nums.size()){
            if (i > 0 && tmp.front() == nums[i - 1]){
                tmp.pop_front();
            }
            while (!tmp.empty() && tmp.back() < nums[j]){
                tmp.pop_back();
            }
            tmp.push_back(nums[j]);
            if (i >= 0){
                res.push_back(tmp.front());
            }
            i++;
            j++;
        }
        return res;
    }
};
```

## 59-II 队列的最大值
请定义一个队列并实现函数 `max_value` 得到队列里的最大值，要求函数 `max_value`、`push_back` 和 `pop_front` 的均摊时间复杂度都是O(1)。

若队列为空，`pop_front` 和 `max_value` 需要返回 -1

**示例 1：**
```
输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]
```
**示例 2：**
```
输入: 
["MaxQueue","pop_front","max_value"]
[[],[],[]]
输出: [null,-1,-1]
```

**限制：**
```
1 <= push_back,pop_front,max_value的总操作数 <= 10000
1 <= value <= 10^5
```

> 思路：双向队列实现单调队列
```C++
class MaxQueue {
public:
    queue<int> q;
    deque<int> tmp;
    MaxQueue() {
        
    }
    
    int max_value() {
        if (!tmp.empty()){
            return tmp.front();
        } else {
            return -1;
        }
    }
    
    void push_back(int value) {
        q.push(value);
        while (!tmp.empty() && tmp.back() < value){
            tmp.pop_back();
        }
        tmp.push_back(value);
    }
    
    int pop_front() {
        if (q.empty()) return -1;
        if (q.front() == tmp.front()){
            tmp.pop_front();
        }
        int res = q.front();
        q.pop();
        return res;        
    }
};
```

## 60 n 个骰子的点数
把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

**示例 1:**
```
输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
```
**示例 2:**
```
输入: 2
输出: [0.02778,0.05556,0.08333,0.11111,0.13889,0.16667,0.13889,0.11111,0.08333,0.05556,0.02778]
```
> 思路：动态规划
f(2,7) = f(1, 7-1) + f(1, 7-2) + ... + f(1, 7-6)
![](https://pic.leetcode-cn.com/1614960989-tpJNRQ-Picture2.png)

目前这个版本还存在多申请一部分空间的问题。
```C++
class Solution {
public:
    vector<double> dicesProbability(int n) {
        vector<double> dp (7, 1.0/6);
        for (int i = 2; i <= n; i++){
            vector<double> tmp (6 * i + 1, 0);
            for (int j = i; j <= 6 * i; j++){
                for (int k = 1; k <= 6; k++){
                    if (j - k > 0){
                        if (j - k <= 6 * (i - 1)){
                            tmp[j] = tmp[j] + dp[j - k] * (1.0 / 6);
                        }                         
                    } else {
                        break;
                    }
                    
                }
            }
            dp = tmp;
        }
        return vector<double> (dp.begin() + n, dp.end());
    }
};
```

修改索引，解决申请多余空间的问题。
```C++
class Solution {
public:
    vector<double> dicesProbability(int n) {
        vector<double> dp (6, 1.0/6);
        for (int i = 2; i <= n; i++){
            vector<double> tmp (5 * i + 1, 0);
            for (int j = 0; j <= 5 * i; j++){
                for (int k = 0; k < 6; k++){
                    if (j - k >= 0){
                        if (j - k <= 5 * (i - 1))
                        tmp[j] = tmp[j] + dp[j - k] * (1.0 / 6);
                    } else {
                        break;
                    }
                }
            }
            dp = tmp;
        }
        return dp;
    }
};
```

## 61 扑克牌中的数字
从若干副扑克牌中随机抽 5 张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

**示例 1:**
```
输入: [1,2,3,4,5]
输出: True
``` 

**示例 2:**
```
输入: [0,0,1,2,5]
输出: True
```

> 思路：集合
![](https://pic.leetcode-cn.com/df03847e2d04a3fcb5649541d4b6733fb2cb0d9293c3433823e04935826c33ef-Picture1.png)

```C++
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        set<int> tmp;
        int mi = 14, ma = 0;
        for (int i = 0; i < nums.size(); i++){
            if (nums[i] == 0) continue;
            if (tmp.find(nums[i]) != tmp.end()) return false;
            if (nums[i] < mi) mi = nums[i];
            if (nums[i] > ma) ma = nums[i];
            tmp.insert(nums[i]);
        }
        return ma - mi < 5;
    }
};
```

## 62 圆圈中最后剩下的数字
0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

**示例 1：**
```
输入: n = 5, m = 3
输出: 3
```
**示例 2：**
```
输入: n = 10, m = 17
输出: 2
```

**限制：**
```
1 <= n <= 10^5
1 <= m <= 10^6
```

> 思路：约瑟夫环

这个问题实际上是约瑟夫问题，这个问题描述是
> N 个人围成一圈，第一个人从 1 开始报数，报M的将被杀掉，下一个人接着从 1 开始报。如此反复，最后剩下一个，求最后的胜利者。

![](https://pic.leetcode-cn.com/d7768194055df1c3d3f6b503468704606134231de62b4ea4b9bdeda7c58232f4-约瑟夫环1.png)
![](https://pic.leetcode-cn.com/68509352d82d4a19678ed67a5bde338f86c7d0da730e3a69546f6fa61fb0063c-约瑟夫环2.png)

推出递推公式: f(n,m) = [f(n-1, m) + m] % n
```C++
class Solution {
public:
    int lastRemaining(int n, int m) {
        int res;
        for (int i = 1; i <= n; i++){
            res = (res + m) % i;
        }
        return res;
    }
};
```

## 63 股票的最大利润
假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

**示例 1:**
```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```
**示例 2:**
```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```
> 思路：动态规划
记录每一个位置为卖出位置的最大收益，记录最大值。  

1. 利用辅助数组
```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.size() == 0) return 0;
        vector<int> dp(prices.size(), 0);
        for (int i = 1; i < prices.size(); i++){
            if ((dp[i-1] + prices[i] - prices[i-1]) > 0){
                dp[i] = dp[i-1] + prices[i] - prices[i-1];
            }
        }
        return *max_element(dp.begin(), dp.end());
    }
};
```

2. 直接创建临时变量即可
```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.size() == 0) return 0;
        int ma = 0, tmp = 0;
        for (int i = 1; i < prices.size(); i++){
            if ((tmp + prices[i] - prices[i-1]) > 0){
                tmp = tmp + prices[i] - prices[i-1];
            } else {
                tmp = 0;
            }
            if (tmp > ma) {
                ma = tmp;
            }
        }
        return ma;
    }
};
```


## 64 求 1+2+3+..+n
求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
**示例 1：**
```
输入: n = 3
输出: 6
```
**示例 2：**
```
输入: n = 9
输出: 45
```

> 思路1：递归
```C++
class Solution {
public:
    int sumNums(int n) {
        if (n == 1) return 1;
        return n + sumNums(n - 1);
    }
};
```

> 思路2：sizeof 实现乘法操作
```C++
class Solution {
public:
    int sumNums(int n) {
        bool a[n][n+1];
        return sizeof(a)>>1;
    }
};
```

## 65 不用加减乘除做加法
写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

**示例:**
```
输入: a = 1, b = 1
输出: 2
```
**提示：**
```
a, b 均可能是负数或 0
结果不会溢出 32 位整数
```

> 思路：位运算  

不进位结果 + 进位结果
```C++
class Solution {
public:
    int add(int a, int b) {
        if (b == 0) {
            return a;
        }
        return add(a ^ b, (unsigned int)(a & b) << 1);
    }
};
```

## 66 构建乘积数组
给定一个数组 `A[0,1,…,n-1]`，请构建一个数组 `B[0,1,…,n-1]`，其中 `B[i]` 的值是数组 `A` 中除了下标 `i` 以外的元素的积, 即 `B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]`。不能使用除法。

**示例:**
```
输入: [1,2,3,4,5]
输出: [120,60,40,30,24]
```

> 思路：分别计算左边和右边两半部分乘积
```C++
class Solution {
public:
    vector<int> constructArr(vector<int>& a) {
        vector<int> left (a.size(), 1);
        vector<int> right (a.size(), 1);
        vector<int> res (a.size(), 1);
        for (int i = 1; i < a.size(); i++){
            left[i] = a[i - 1] * left[i - 1];
        }
        for (int i = a.size() - 2; i >= 0; i--){
            right[i] = a[i + 1] * right[i + 1];
        }
        for (int i = 0; i < a.size(); i++){
            res[i] = left[i] * right[i];
        }
        return res;
    }
};
```

## 68-I 二叉搜索树的最近公共祖先
给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  `root = [6,2,8,0,4,7,9,null,null,3,5]`

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/binarysearchtree_improved.png)

**示例 1:**
```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。
```
**示例 2:**
```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
```

> 思路1: 迭代

最近公共祖先的定义：设节点 `root` 为节点 `p, q` 的某公共祖先，若其左子节点 `root.left` 和右子节点 `root.right` 都不是 `p, q` 的公共祖先，则称 `root` 是“最近的公共祖先”。

```C++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        while (1) {
            if (root -> val > p -> val && root -> val > q -> val) {
                root = root -> left;
            } else if (root -> val < p -> val && root -> val < q -> val) {
                root = root -> right;
            } else {
                break;
            }
        }
        return root;
    }
};
```

> 思路2: 递归
```C++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        while (1) {
            if (root -> val > p -> val && root -> val > q -> val) {
                root = root -> left;
            } else if (root -> val < p -> val && root -> val < q -> val) {
                root = root -> right;
            } else {
                break;
            }
        }
        return root;
    }
};
```

## 68-II 二叉树的最近公共祖先
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉树:  `root = [3,5,1,6,2,0,8,null,null,7,4]`

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/15/binarytree.png)

**示例 1:**
```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
```
**示例 2:**
```
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
```

**说明:**
+ 所有节点的值都是唯一的。
+ p、q 为不同节点且均存在于给定的二叉树中。

> 思路：递归
```C++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (root == nullptr) return nullptr;  // 如果树为空，直接返回null
        if (root == p || root == q) return root;  // 如果 p和q中有等于 root的，那么它们的最近公共祖先即为root（一个节点也可以是它自己的祖先）
        TreeNode* left = lowestCommonAncestor(root -> left, p, q);  // 递归遍历左子树，只要在左子树中找到了p或q，则先找到谁就返回谁
        TreeNode* right = lowestCommonAncestor(root -> right, p, q);  // 递归遍历右子树，只要在右子树中找到了p或q，则先找到谁就返回谁
        if (left == nullptr) return right;  // 如果在左子树中 p和 q都找不到，则 p和 q一定都在右子树中，右子树中先遍历到的那个就是最近公共祖先（一个节点也可以是它自己的祖先）
        else if (right == nullptr) return left;  // 否则，如果 left不为空，在左子树中有找到节点（p或q），这时候要再判断一下右子树中的情况，如果在右子树中，p和q都找不到，则 p和q一定都在左子树中，左子树中先遍历到的那个就是最近公共祖先（一个节点也可以是它自己的祖先）
        else return root;  //否则，当 left和 right均不为空时，说明 p、q节点分别在 root异侧, 最近公共祖先即为 root
    }
    }
};
```

> 思路2: 哈希表
```C++
class Solution {
public:
    unordered_map<int, TreeNode*> parent;
    unordered_map<int, bool> visited;
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        parent[root -> val] = nullptr;
        dfs(root);
        while (p != nullptr) {
            visited[p -> val] = true;
            p = parent[p -> val];
        }
        while (q != nullptr) {
            if (visited[q -> val] == true) {
                return q;
            }
            q = parent[q -> val];
        }
        return nullptr;
    }

    void dfs(TreeNode* root) {
        if (root -> left != nullptr) {
            parent[root -> left -> val] = root;
            dfs(root -> left);
        }
        if (root -> right != nullptr) {
            parent[root -> right -> val] = root;
            dfs(root -> right);
        }
    }
};
```


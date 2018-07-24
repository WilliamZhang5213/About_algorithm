#--coding:utf-8
import numpy as np
import sys

def package():
    #背包问题，动态规划解法
    #定义重量
    weight = {}
    weight['water'] = 3
    weight['book'] = 1
    weight['food'] = 2
    weight['jacket'] = 2
    weight['camera'] = 1

    #定义价值
    worth = {}
    worth['water'] = 10
    worth['book'] = 3
    worth['food'] = 9
    worth['jacket'] = 5
    worth['camera'] = 6

    #存放行标对应的物品名
    table_name = {}
    table_name[0] = 'water'
    table_name[1] = 'book'
    table_name[2] = 'food'
    table_name[3] = 'jacket'
    table_name[4] = 'camera'

    #创建矩阵，保存价值表， 6表示背包最大重量
    table = np.zeros((len(weight), 6))

    #创建矩阵，用来保存每个单元格中的价值是如何得到的（物品名）
    table_class = np.zeros((len(weight), 6), dtype = np.dtype((np.str_, 500)))

    for i in range(0, len(weight)):
        for j in range(0, 6):
            #获取重量
            this_weight = weight[table_name[i]]
            #获得价值
            this_worth = worth[table_name[i]]
            #获取上一个单元格（i - 1， j）的值
            if i > 0:
                before_worth = table[i - 1, j]
                #获取（i - 1， j - 重量）
                tmp = 0
                if this_weight <= j:
                    tmp = table[i - 1, j - this_weight]
                #（i - 1，j - this_weight）+ 求当前商品价值
                #判断this_worth能不能用，即重量有没有超标，如果重量超标了是不能加的
                synthesize_worth = 0
                if this_weight - 1 <= j:
                    synthesize_worth = this_worth + tmp
                
                if synthesize_worth > before_worth:
                    table[i, j] = synthesize_worth
                    if tmp == 0:
                        #他自己就超过了
                        table_class[i][j] = table_name[i]
                    else:
                        # 他自己和(i-1,j-this_weight)
                        table_class[i][j] = table_name[i] + "," + table_class[i - 1][j - this_weight]
                else:
                    table[i,j]=before_worth
                    table_class[i][j]=table_class[i-1][j]

            else:
               #没有（i - 1，j）那更没有（i - 1，j - 重量），就等于当前商品价值，或者重量不够，是0
               if this_weight - 1 <= j:
                   table[i, j] = this_worth
                   table_class[i][j] = table_name[i]
    print table
    print "------------------------"
    print table_class

def FindGreatestSum(array):
    #找出一维数组最大序列的和
    dp = array[:]
    for i in range(len(array)):
        if i == 0:
            dp[i] = array[i]
        else:
            dp[i] = max(dp[i - 1] + array[i], array[i])
    max_sum = max(dp)
    return max_sum

def MergeConsequentMaxNum(v, i, j, N):
    merge_array = [0]*N
    for p in range(i, j + 1):
        for q in range(N):
            merge_array[q] += v[p][q]
    max_sum = FindGreatestSum(merge_array)
    print merge_array, max_sum
    return max_sum

def GetMaxSubMatrix(v, N):
    #已知矩阵的大小定义为矩阵中所有元素的和。给定一个矩阵，你的任务是找到最大的非空(大小至少是1 * 1)子矩阵。 
    '''
        N: N*N array
        v: input array
    '''
    Max = 0
    for i in range(N):
        for j in range(i, N):
            Max = max(MergeConsequentMaxNum(v, i, j, N), Max)
    return Max

if __name__ == "__main__":
    # package()

    N = 4
    v = [[0, -2, -7, 0],[9, 2, -6, 2],[-4, 1, -4, 1],[-1, 8, 0, -2]]
    print GetMaxSubMatrix(v, N)

